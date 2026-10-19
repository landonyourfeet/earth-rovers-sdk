import asyncio
import collections
import contextlib
import json
import logging
import os
import re
import signal
import tempfile
import time
from typing import Optional

from dotenv import load_dotenv
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import async_playwright

load_dotenv()

logger = logging.getLogger("browser_service")

# Configuration from environment variables with defaults
FORMAT = os.getenv("IMAGE_FORMAT", "jpeg")
QUALITY = float(os.getenv("IMAGE_QUALITY", "0.8"))
# ★ OKCREAL (Sep 5, 2026): feed frames scaled to FEED_WIDTH px wide at a lower
#   JPEG quality — latency work, see captureFrameAsBase64 in basicVideoCall.js.
FEED_QUALITY = float(os.getenv("FEED_JPEG_QUALITY", "0.55"))
FEED_WIDTH = int(os.getenv("FEED_WIDTH", "640"))

if FORMAT not in ["png", "jpeg", "webp"]:
    raise ValueError("Invalid image format. Supported formats: png, jpeg, webp")

if QUALITY < 0 or QUALITY > 1:
    raise ValueError("Invalid image quality. Quality should be between 0 and 1")

if FEED_QUALITY < 0 or FEED_QUALITY > 1:
    raise ValueError("Invalid feed quality. Quality should be between 0 and 1")

SDK_PAGE_URL = os.getenv("SDK_PAGE_URL", "http://127.0.0.1:8000/sdk")


# ★ OKCREAL (Oct 19, 2026 — RV2, Landon: "myrover 2 is live and connected to frodo bots but not being see in connect").
#   Connect could only say "could not join Volt Radon Lima — BrowserContext.new_page: Target page, context or browser has been
#   closed": the first line of Playwright's error. Why Chrome died, and what the rover page said while it tried to join, were
#   thrown away. These read the container's own limits (memory, /dev/shm — the two things that kill headless Chrome) and keep
#   the lines that explain a failed join, so /status and Connect can say the real cause.
def _read_int(path):
    try:
        with open(path) as f:
            v = f.read().strip()
        return None if v in ("", "max") else int(v)
    except Exception:
        return None


def host_facts() -> dict:
    out = {"cpus": os.cpu_count()}
    try:
        st = os.statvfs("/dev/shm")
        out["shm_mb"] = round(st.f_blocks * st.f_frsize / 1048576)
        out["shm_free_mb"] = round(st.f_bavail * st.f_frsize / 1048576)
    except Exception:
        out["shm_mb"] = None
    limit = _read_int("/sys/fs/cgroup/memory.max")
    if limit is None:
        limit = _read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    out["mem_limit_mb"] = round(limit / 1048576) if limit and limit < (1 << 50) else None
    used = _read_int("/sys/fs/cgroup/memory.current")
    if used is None:
        used = _read_int("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    out["mem_used_mb"] = round(used / 1048576) if used else None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    out["mem_available_mb"] = round(int(line.split()[1]) / 1024)
    except Exception:
        pass
    # ★ RV2f (Oct 19, 2026): myRover2's Chrome could not start a thread ("pthread_create: Resource temporarily unavailable") — the
    #   container's process/thread limit and how much of it is in use
    pm = _read_int("/sys/fs/cgroup/pids.max")
    if pm is None:
        pm = _read_int("/sys/fs/cgroup/pids/pids.max")
    pc = _read_int("/sys/fs/cgroup/pids.current")
    if pc is None:
        pc = _read_int("/sys/fs/cgroup/pids/pids.current")
    out["pids_max"], out["pids_current"] = pm, pc
    try:
        import resource
        soft = resource.getrlimit(resource.RLIMIT_NPROC)[0]
        out["nproc_limit"] = None if soft == resource.RLIM_INFINITY or soft < 0 else int(soft)
    except Exception:
        out["nproc_limit"] = None
    try:
        out["threads"] = sum(len(os.listdir(f"/proc/{d}/task")) for d in os.listdir("/proc") if d.isdigit() and os.path.isdir(f"/proc/{d}/task"))
    except Exception:
        out["threads"] = None
    out["zombies"] = zombie_count()   # RV2g: finished processes nobody reaped — each still holds a PID/thread slot
    return out


def host_facts_text(f: Optional[dict] = None) -> str:
    f = f or host_facts()
    bits = []
    if f.get("mem_limit_mb"):
        bits.append("memory %s MB limit, %s MB used" % (f["mem_limit_mb"], f.get("mem_used_mb", "?")))
    elif f.get("mem_available_mb") is not None:
        bits.append("memory %s MB free" % f["mem_available_mb"])
    if f.get("shm_mb") is not None:
        bits.append("/dev/shm %s MB (%s MB free)" % (f["shm_mb"], f.get("shm_free_mb", "?")))
    lims = [x for x in (f.get("pids_max"), f.get("nproc_limit")) if x]
    if f.get("threads") is not None or lims:
        bits.append("threads %s in use%s" % (f.get("threads", "?") if f.get("threads") is not None else (f.get("pids_current") or "?"),
                                             " (limit %s)" % min(lims) if lims else " (no limit set)"))
    if f.get("zombies"):
        bits.append("%s unreaped (zombie) processes" % f["zombies"])
    return ", ".join(bits) or "no host facts readable"


_CHROME_WHY = re.compile(r"error|fatal|shm|memory|killed|crash|denied|no space|sandbox|signal", re.I)
_PAGE_WHY = re.compile(r"fail|error|abort|denied|token|login|connectionstate|refused|expired|unauthor|invalid|offline", re.I)
_PAGE_STRONG = re.compile(r"rtm|login|token|abort|unauthor|expired|channel|join", re.I)


def _console_line(text: str) -> str:
    # one line per message, without the stack trace or the console's %c styling
    line = str(text or "").split("\n", 1)[0].replace("%c", "")
    return re.sub(r"\s*color:\s*#?[0-9a-z]+;\s*font-weight:\s*\w+;?", "", line, flags=re.I).strip()[:240]


# ★ OKCREAL (Oct 19, 2026 — RV2c): myRover2's card read "trying to join … again now" for 42 minutes — one launch hung (a page that
#   never opened after Chrome died). No launch may hang: past this many seconds it is cut, recorded with the step it was stuck on,
#   the browser and the Playwright driver are thrown away, and the next try starts clean.
LAUNCH_TIMEOUT_S = float(os.getenv("BROWSER_LAUNCH_TIMEOUT_S", "120"))


# ★ OKCREAL (Oct 19, 2026 — RV2g): myRover2's container sat at "threads 938 in use (limit 1000)". The SDK runs as PID 1 there
#   (Dockerfile CMD), so every Chrome process orphaned by a closed browser is re-parented to it — and Python never reaps children it did
#   not start, so each one stays a zombie holding a PID/thread slot until the container can start no thread at all
#   ("pthread_create: Resource temporarily unavailable", "Did not receive ping from zygote child"). Rover 2 retries a failing join every
#   few seconds, so it filled first. The SDK now makes itself the subreaper of what it starts (when it is not PID 1) and reaps finished
#   orphaned Chrome processes.
def become_subreaper() -> bool:
    try:
        import ctypes
        return ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0   # PR_SET_CHILD_SUBREAPER
    except Exception:
        return False


def _proc_stat(d: str):
    with open(f"/proc/{d}/stat") as f:
        st = f.read()
    comm = st[st.index("(") + 1:st.rindex(")")]
    rest = st[st.rindex(")") + 2:].split()
    return comm, rest[0], int(rest[1])


def reap_orphans() -> int:
    """Reap finished Chrome processes that were re-parented to this process (zombies). Only browser processes — never the
    Playwright driver or anything asyncio is waiting on."""
    me, n = os.getpid(), 0
    try:
        entries = os.listdir("/proc")
    except Exception:
        return 0
    for d in entries:
        if not d.isdigit():
            continue
        try:
            comm, state, ppid = _proc_stat(d)
        except Exception:
            continue
        if state == "Z" and ppid == me and re.search(r"chrom|crashpad|headless", comm, re.I):
            try:
                if os.waitpid(int(d), os.WNOHANG)[0]:
                    n += 1
            except ChildProcessError:
                pass
            except Exception:
                pass
    return n


def zombie_count() -> int:
    n = 0
    try:
        for d in os.listdir("/proc"):
            if d.isdigit():
                try:
                    if _proc_stat(d)[1] == "Z":
                        n += 1
                except Exception:
                    pass
    except Exception:
        pass
    return n


def _descendants(root: int) -> list:
    """Every process started under root (Linux /proc) — the Playwright driver and the browsers it launched."""
    kids = {}
    try:
        for d in os.listdir("/proc"):
            if not d.isdigit():
                continue
            try:
                with open(f"/proc/{d}/stat") as f:
                    ppid = int(f.read().rsplit(")", 1)[1].split()[1])
                kids.setdefault(ppid, []).append(int(d))
            except Exception:
                continue
    except Exception:
        return []
    out, todo = [], [root]
    while todo:
        for k in kids.get(todo.pop(), []):
            out.append(k)
            todo.append(k)
    return out


def sdk_page_error(status: int, body: str, token_error: Optional[str]) -> str:
    """★ OKCREAL (Oct 19, 2026 — RV2b): the rover page /sdk answered an error instead of the join page (FrodoBots refused the
    bot's token), so there is no #join button and Playwright only said "Page.click: Timeout 30000ms exceeded". Say what the
    page answered and which FrodoBots call failed."""
    detail = ""
    try:
        parsed = json.loads(body or "")
        if isinstance(parsed, dict) and isinstance(parsed.get("detail"), str):
            detail = parsed["detail"]
    except (TypeError, ValueError):
        pass
    if not detail:
        detail = re.sub(r"<[^>]*>", " ", str(body or ""))
        detail = re.sub(r"\s+", " ", detail).strip()
    msg = f"the rover page /sdk answered {status}" + (f": {detail[:160]}" if detail else "")
    if token_error:
        msg += f" — {token_error}"
    return msg[:400]


# ★ OKCREAL (Oct 19, 2026 — RV2d): myRover2's Chrome starts but a page never opens ("hung for 120 s while opening a browser page"),
#   and Playwright says nothing about why. Chrome writes its own log to a file (--enable-logging --log-file); after a failed or hung
#   launch the lines that say why (FATAL, a crash, shared memory, the GPU or zygote, out of memory) join the diagnosis.
_CHROME_LOG_NOISE = re.compile(r":CONSOLE[:(]|should be created through the zygote|dbus|Failed to connect to the bus|fontconfig|gcm|google_apis|ALSA|pulse|UPower|libva|vaapi|"   # page console lines are told as "page said"
                               r"ssl_client_socket|registration_request|voice_transcription|on_device_model|optimization_guide", re.I)
_CHROME_LOG_WHY = re.compile(r"FATAL|Check failed|crash|shared.?memory|/dev/shm|No space|zygote|sandbox|renderer|gpu|out of memory|"
                             r"OOM|Bus error|SIGBUS|SIGSEGV|SIGKILL|ERROR", re.I)


def chrome_log_lines(text: str) -> list:
    """The lines of Chrome's own log that explain a failed launch (noise like D-Bus left out), shortened, newest last."""
    out = []
    for raw in str(text or "").splitlines()[-400:]:
        line = raw.strip()
        if not line or _CHROME_LOG_NOISE.search(line) or not _CHROME_LOG_WHY.search(line):
            continue
        line = re.sub(r"^\[[^\]]*\]\s*", "", line)   # [pid:tid:date:LEVEL:file(line)] prefix — keep the words
        line = line[:200]
        if line not in out:
            out.append(line)
    return out[-3:]


def launch_diagnosis(err_text: str, console: list, chrome_log: Optional[list] = None) -> str:
    lines = str(err_text or "").split("\n")
    first = lines[0].strip() or "browser launch failed"
    parts = [first]
    chrome = [l.strip() for l in lines[1:] if l.strip() and _CHROME_WHY.search(l)]
    if chrome:
        parts.append("chrome: " + " | ".join(chrome[-2:])[:200])
    browser_down = bool(re.search(r"closed|crash|target|hung", first, re.I))
    if chrome_log and browser_down:   # only when Chrome itself failed — a rover-link failure is told by what the page said
        parts.append("chrome log: " + " | ".join(chrome_log[-3:])[:260])
    said = []
    for c in console:
        if _PAGE_WHY.search(c) and c not in said:
            said.append(c)
    # the rover link's own words (Agora RTM login, token, channel) outrank a stray resource that would not load
    strong = [c for c in said if _PAGE_STRONG.search(c)]
    weak = [c for c in said if c not in strong]
    pick = (strong[-2:] + weak[-1:]) if strong else weak[-3:]
    if pick:
        parts.append("page said: " + " | ".join(pick)[:260])
    if browser_down or chrome:
        parts.append("host: " + host_facts_text())
    return " · ".join(parts)[:600]


class BrowserService:
    def __init__(self):
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._ready = False
        self._lock = None
        self._lock_loop = None
        self.last_error = None
        self.last_error_at: Optional[float] = None
        self.launching = False
        # RV2e: Chrome without its zygote — set by CHROME_NO_ZYGOTE=1, or once Chrome's own log says its zygote failed (myRover2:
        #   "Did not receive ping from zygote child"); kept for the life of this SDK process
        self.no_zygote = os.getenv("CHROME_NO_ZYGOTE", "") == "1"
        # RV2f: a lean Chrome (no GPU process, one raster thread, few renderers) — CHROME_LEAN=1, or once Chrome's own log says a
        #   thread or process could not be started ("pthread_create: Resource temporarily unavailable")
        self.lean = os.getenv("CHROME_LEAN", "") == "1"
        self.reaped_total = 0   # RV2g: orphaned Chrome processes reaped
        # RV2c: the step a join is on right now, and when it began (a hung launch is cut at LAUNCH_TIMEOUT_S)
        self.launch_step: Optional[str] = None
        self.launch_started_at: Optional[float] = None
        # RV2b: set by main.py when FrodoBots refuses the bot's token ("FrodoBots /sdk/token answered 404 for bot 'x'")
        self.token_error: Optional[str] = None
        # RV2: what the rover page printed while it was joining (Agora RTM login, channel join)
        self._console = collections.deque(maxlen=40)
        # has_rear_camera() cache: (value, monotonic expiry). A positive
        # result holds for the page's lifetime; a negative one only briefly,
        # since the rear track can subscribe late after the Agora join.
        self._rear_camera: Optional[tuple[bool, float]] = None
        # Large enough for the legacy front/rear/map element captures without
        # paying for an 8.3-megapixel headless render surface on every frame.
        self._viewport = {"width": 1920, "height": 1200}

    def _get_lock(self) -> asyncio.Lock:
        # On Python 3.9 an asyncio.Lock binds to the loop present when it is
        # constructed. This service is instantiated at import time — before
        # hypercorn creates its serving loop — so the lock must be created
        # (and recreated after a --reload loop swap) inside the running loop.
        running_loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not running_loop:
            self._lock = asyncio.Lock()
            self._lock_loop = running_loop
        return self._lock

    @property
    def is_ready(self) -> bool:
        return bool(
            self._ready
            and self._page
            and not self._page.is_closed()
            and self._browser
            and self._browser.is_connected()
        )

    async def ensure_page(self):
        # Lock-free fast path: concurrent /control, /feed and /v2 calls must
        # not serialize on the init lock once the page is up.
        if self.is_ready:
            return self._page
        async with self._get_lock():
            if self.is_ready:
                return self._page
            await self._teardown()
            try:
                await asyncio.wait_for(self._launch(), timeout=LAUNCH_TIMEOUT_S)
            except asyncio.TimeoutError:
                step = self.launch_step or "starting the browser"
                clog = self._chrome_log_tail()
                retry = self._zygote_check(clog)
                self.last_error = launch_diagnosis(
                    f"the browser launch hung for {int(LAUNCH_TIMEOUT_S)} s while {step} — closed it to try again"
                    + ((" with a lean Chrome (no GPU process, few threads)" if self.lean else " without Chrome's zygote (--no-zygote)") if retry else ""),
                    list(self._console),
                    clog,
                )
                self.last_error_at = time.time()
                logger.error("Launch diagnosis: %s", self.last_error)
                await self._hard_reset()
                raise RuntimeError(self.last_error)
            return self._page

    def _zygote_check(self, clog: list) -> bool:
        # ★ RV2f: the container ran out of threads/processes → a lean Chrome from now on (and without the zygote, which needs them too)
        if not self.lean and any(re.search(r"pthread_create|Resource temporarily unavailable|fork\(\) failed|Cannot allocate memory", l, re.I) for l in (clog or [])):
            self.lean = True
            self.no_zygote = True
            logger.warning("The container ran out of threads for Chrome — launching a lean Chrome (--disable-gpu, few renderers, --no-zygote) from now on")
            return True
        # ★ RV2e: Chrome's zygote failing in this container (myRover2) → every later launch goes without it. True when this changed.
        if not self.no_zygote and any(re.search(r"did not receive ping from zygote|zygote.*(fail|died|crash|NOTREACHED|Check failed)|(fail|NOTREACHED|Check failed).*zygote", l, re.I) for l in (clog or [])):
            self.no_zygote = True
            logger.warning("Chrome's zygote failed here — launching without it (--no-zygote) from now on")
            return True
        return False

    def _chrome_log_tail(self) -> list:
        path = getattr(self, "_chrome_log_path", None)
        if not path:
            return []
        try:
            with open(path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 64000))
                return chrome_log_lines(f.read().decode("utf-8", "replace"))
        except Exception:
            return []

    async def _hard_reset(self):
        # a hung launch: close what we can (never waiting long on a dead browser), then drop the driver so the next try starts clean.
        #   A Chrome the driver was still starting is not closed by stopping the driver, so whatever this process started is ended too.
        kids = _descendants(os.getpid())
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self._teardown(), timeout=10)
        self._page = self._context = self._browser = None
        self._ready = False
        if self._playwright:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._playwright.stop(), timeout=15)
            self._playwright = None
        for pid in kids:
            with contextlib.suppress(Exception):
                os.kill(pid, signal.SIGKILL)
        self.launching = False
        self.launch_step = None

    async def _launch(self):
        self._ready = False
        self.launching = True
        self.launch_started_at = time.time()
        self.launch_step = "starting the Playwright driver"
        self._console.clear()
        try:
            if self._playwright is None:
                self._playwright = await async_playwright().start()
            self.launch_step = "launching Chrome"

            self._chrome_log_path = os.path.join(tempfile.gettempdir(), f"okc-chrome-{os.getpid()}.log")
            with contextlib.suppress(Exception):
                os.remove(self._chrome_log_path)
            launch_kwargs = {
                "headless": True,
                "args": [
                    "--enable-logging",
                    f"--log-file={self._chrome_log_path}",
                    "--v=0",
                    *(["--no-zygote"] if self.no_zygote else []),
                    *(["--disable-gpu", "--renderer-process-limit=2", "--num-raster-threads=1", "--disable-extensions"] if self.lean else []),
                    "--ignore-certificate-errors",
                    "--no-sandbox",
                    "--autoplay-policy=no-user-gesture-required",
                    "--use-fake-ui-for-media-stream",
                    "--disable-application-cache",
                    "--disk-cache-size=0",
                ],
            }
            executable_path = os.getenv("CHROME_EXECUTABLE_PATH") or None
            if executable_path:
                self._browser = await self._playwright.chromium.launch(
                    executable_path=executable_path, **launch_kwargs
                )
                logger.info("Using browser from CHROME_EXECUTABLE_PATH")
            else:
                # Prefer installed Google Chrome: it ships the H.264/AAC
                # codecs some rover streams need; Playwright's open-source
                # Chromium does not and decodes those streams as 0x0.
                try:
                    self._browser = await self._playwright.chromium.launch(
                        channel="chrome", **launch_kwargs
                    )
                    logger.info("Using installed Google Chrome (all codecs)")
                except PlaywrightError:
                    if not os.path.exists(
                        self._playwright.chromium.executable_path
                    ):
                        raise RuntimeError(
                            "No usable browser found. Run: python -m"
                            " playwright install chromium (or install Google"
                            " Chrome, or set CHROME_EXECUTABLE_PATH)"
                        )
                    self._browser = await self._playwright.chromium.launch(
                        **launch_kwargs
                    )
                    logger.info(
                        "Using Playwright's bundled Chromium (no Google"
                        " Chrome found). If video frames stay empty, the"
                        " stream may need H.264: install Chrome or set"
                        " CHROME_EXECUTABLE_PATH to a codec-capable browser"
                    )
            self.launch_step = "opening a browser page"
            self._context = await self._browser.new_context(
                viewport=self._viewport,
                extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
            )
            self._page = await self._context.new_page()
            self._page.on(
                "console", lambda m: self._console.append(_console_line(f"{m.type}: {m.text}"))
            )
            self._page.on(
                "pageerror", lambda e: self._console.append(_console_line(f"pageerror: {e}"))
            )
            self._page.on("crash", lambda *_: self._console.append("error: page crashed"))
            self.launch_step = "loading the rover page /sdk"
            page_resp = await self._page.goto(SDK_PAGE_URL, wait_until="domcontentloaded")
            if page_resp is not None and page_resp.status >= 400:
                try:
                    page_body = await page_resp.text()
                except Exception:
                    page_body = ""
                raise RuntimeError(sdk_page_error(page_resp.status, page_body, self.token_error))
            self.launch_step = "pressing Join on the rover page"
            await self._page.click("#join")
            self.launch_step = "waiting for the rover link (RTM login)"
            # Control and telemetry must remain available when a camera is
            # offline. Wait for RTM readiness, not for a video DOM element.
            await self._page.wait_for_function(
                "() => typeof window.sendMessage === 'function'"
                " && window.rtmReady === true",
                timeout=15000,
            )

            call = f"""() => {{
                window.initializeImageParams({{
                    imageFormat: "{FORMAT}",
                    imageQuality: {QUALITY}
                }});
            }}"""
            await self._page.evaluate(call)
            self._ready = True
            self.last_error = None
            logger.info("Headless browser connected to %s", SDK_PAGE_URL)
        except Exception as e:
            clog = self._chrome_log_tail()
            retry = self._zygote_check(clog)
            self.last_error = launch_diagnosis(str(e) + ((" — the next try runs a lean Chrome (no GPU process, few threads)" if self.lean else " — the next try runs Chrome without its zygote (--no-zygote)") if retry else ""), list(self._console), clog)
            self.last_error_at = time.time()
            logger.error("Error initializing browser: %s", e)
            logger.error("Launch diagnosis: %s", self.last_error)
            await self._teardown()
            # A failed Playwright transport cannot recover by reusing the same
            # driver instance. Recreate it on the next warm-up attempt.
            if self._playwright:
                try:
                    await self._playwright.stop()
                except Exception:
                    pass
                self._playwright = None
            raise
        finally:
            self.launching = False   # launch_step keeps the last step: a hung launch is reported by where it stuck

    def reap(self) -> int:
        n = reap_orphans()
        self.reaped_total += n
        return n

    async def _teardown(self):
        self._ready = False
        self._rear_camera = None
        for target in (self._page, self._context, self._browser):
            if target:
                try:
                    await target.close()
                except Exception:
                    pass
        self._page = None
        self._context = None
        self._browser = None

    async def _invalidate(self, failed_page):
        """Tear down only if the failed page is still the active generation."""
        async with self._get_lock():
            if self._page is failed_page:
                await self._teardown()

    async def _run(self, fn, *, retry_on_disconnect: bool = True):
        page = await self.ensure_page()
        try:
            return await fn(page)
        except PlaywrightError as e:
            disconnected = page.is_closed() or not (
                self._browser and self._browser.is_connected()
            )
            if not disconnected:
                # JavaScript/application errors are not browser crashes. Do not
                # disrupt feed/control calls that are sharing a healthy page.
                raise
            logger.warning("Browser disconnected (%s); relaunching", e)
            await self._invalidate(page)
            if not retry_on_disconnect:
                raise
            page = await self.ensure_page()
            return await fn(page)

    async def warmup(self, max_attempts: int = 5):
        delay = 2
        for attempt in range(1, max_attempts + 1):
            try:
                await self.ensure_page()
                return True
            except Exception as e:
                logger.warning(
                    "Browser warm-up attempt %s/%s failed: %s",
                    attempt,
                    max_attempts,
                    e,
                )
                if attempt < max_attempts:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30)
        logger.warning(
            "Browser warm-up gave up after %s attempts;"
            " it will initialize lazily on the next request",
            max_attempts,
        )
        return False

    async def capture_screenshots(self, elements: list) -> dict[str, bytes]:
        element_map = {"front": "#player-1000", "rear": "#player-1001", "map": "#map"}

        async def capture(page):
            screenshots = {}
            for name in elements:
                if name not in element_map:
                    logger.warning("Invalid element name: %s", name)
                    continue
                locator = page.locator(element_map[name])
                if await locator.count() == 0:
                    logger.warning("Element %s not found", element_map[name])
                    continue
                start_time = time.time()
                image = await locator.screenshot(type="png", timeout=5000)
                elapsed_ms = (time.time() - start_time) * 1000
                logger.info("Screenshot for %s took %.2f ms", name, elapsed_ms)
                screenshots[name] = image
            return screenshots

        return await self._run(capture)

    async def take_screenshot(self, video_output_folder: str, elements: list):
        """Backward-compatible wrapper; new API consumers should use bytes."""
        screenshots = await self.capture_screenshots(elements)
        paths = {}
        for name, image in screenshots.items():
            output_path = os.path.join(video_output_folder, f"{name}.png")
            await asyncio.to_thread(self._write_file, output_path, image)
            paths[name] = output_path
        return paths

    @staticmethod
    def _write_file(path: str, content: bytes):
        with open(path, "wb") as output:
            output.write(content)

    async def data(self) -> dict:
        return await self._run(lambda page: page.evaluate("() => window.rtm_data"))

    async def has_rear_camera(self) -> bool:
        # Capability comes from reality, not BOT_TYPE: the rover either
        # publishes a rear video track (uid 1001) or it doesn't. Cached so
        # frame polling doesn't pay a page round trip on every request.
        cached = self._rear_camera
        if cached and (cached[0] or time.monotonic() < cached[1]):
            return cached[0]
        result = bool(
            await self._run(
                lambda page: page.evaluate(
                    "() => !!(typeof remoteUsers !== 'undefined'"
                    " && remoteUsers[1001] && remoteUsers[1001].videoTrack)"
                )
            )
        )
        self._rear_camera = (result, time.monotonic() + 5)
        return result

    async def front(self) -> str:
        return await self._run(
            lambda page: page.evaluate("() => getLastBase64Frame(1000) || null")
        )

    async def rear(self) -> str:
        return await self._run(
            lambda page: page.evaluate("() => getLastBase64Frame(1001) || null")
        )

    async def front_feed(self) -> dict:
        return await self._run(
            lambda page: page.evaluate(
                "([quality, width]) => getFramePacket(1000, 'jpeg', quality, width)",
                [FEED_QUALITY, FEED_WIDTH],
            )
        )

    async def rear_feed(self) -> dict:
        return await self._run(
            lambda page: page.evaluate(
                "([quality, width]) => getFramePacket(1001, 'jpeg', quality, width)",
                [FEED_QUALITY, FEED_WIDTH],
            )
        )

    async def configured_frame(self, view: str) -> dict:
        uid = 1000 if view == "front" else 1001
        return await self._run(
            lambda page: page.evaluate(
                "([uid, format, quality]) => getFramePacket(uid, format, quality)",
                [uid, FORMAT, QUALITY],
            )
        )

    async def send_message(self, message: dict):
        # Non-blocking dispatch: returns once the message is on the wire.
        # Delivery is observed asynchronously via rtm_health().
        return await self._run(
            lambda page: page.evaluate(
                "(message) => window.sendMessage(message)", message
            ),
            retry_on_disconnect=False,
        )

    async def send_message_confirmed(self, message: dict) -> bool:
        """Send and wait for the rover's receipt (hasPeerReceived)."""
        result = await self._run(
            lambda page: page.evaluate(
                "async (message) => await window.sendMessageAwait(message)",
                message,
            ),
            retry_on_disconnect=False,
        )
        return result is True

    async def rtm_health(self) -> Optional[dict]:
        if not self.is_ready:
            return None
        try:
            return await self._run(
                lambda page: page.evaluate(
                    "() => window.rtmHealth ? window.rtmHealth() : null"
                ),
                retry_on_disconnect=False,
            )
        except Exception:
            return None

    async def reset(self):
        """Force a relaunch on next use — rebuilds the page, the Agora
        connections, and the RTM session (for when RTM dies but the page
        itself is still healthy)."""
        async with self._get_lock():
            await self._teardown()

    # ★ OKCREAL (Sep 4, 2026): live stream to the rover speaker.
    async def play_live(self, stream_url: str):
        return await self._run(
            lambda page: page.evaluate(
                "async (u) => await window.playLiveAudioToRover(u)", stream_url
            ),
            retry_on_disconnect=False,
        )

    async def stop_live(self):
        return await self._run(
            lambda page: page.evaluate("async () => await window.stopLiveAudioToRover()"),
            retry_on_disconnect=False,
        )

    async def live_status(self):
        return await self._run(
            lambda page: page.evaluate("() => window.liveAudioStatus()"),
            retry_on_disconnect=False,
        )

    # ★ OKCREAL (Sep 5, 2026): rover microphone tap on/off (see setAudioTap in
    #   static/basicVideoCall.js). Never relaunches the browser for this.
    async def talk_feed(self, b64: str, rate: int = 16000):
        return await self._run(
            lambda page: page.evaluate("async ([b, r]) => await window.talkFeed(b, r)", [b64, rate]),
            retry_on_disconnect=False,
        )

    async def talk_stop(self):
        return await self._run(lambda page: page.evaluate("async () => await window.talkStop()"), retry_on_disconnect=False)

    async def audio_tap(self, on: bool):
        return await self._run(
            lambda page: page.evaluate(
                "async (on) => await window.setAudioTap(on)", bool(on)
            ),
            retry_on_disconnect=False,
        )

    async def speak(self, audio_url: str):
        return await self._run(
            lambda page: page.evaluate(
                "async (audioUrl) => await window.playAudioToRover(audioUrl)",
                audio_url,
            ),
            retry_on_disconnect=False,
        )

    async def close(self):
        async with self._get_lock():
            await self._teardown()
            if self._playwright:
                try:
                    await self._playwright.stop()
                except Exception:
                    pass
                self._playwright = None

    # Backward-compatible alias
    async def close_browser(self):
        await self.close()
