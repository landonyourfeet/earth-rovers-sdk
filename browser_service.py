import asyncio
import collections
import contextlib
import json
import logging
import os
import re
import signal
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


def launch_diagnosis(err_text: str, console: list) -> str:
    lines = str(err_text or "").split("\n")
    first = lines[0].strip() or "browser launch failed"
    parts = [first]
    chrome = [l.strip() for l in lines[1:] if l.strip() and _CHROME_WHY.search(l)]
    if chrome:
        parts.append("chrome: " + " | ".join(chrome[-2:])[:200])
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
    if re.search(r"closed|crash|target", first, re.I) or chrome:
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
                self.last_error = launch_diagnosis(
                    f"the browser launch hung for {int(LAUNCH_TIMEOUT_S)} s while {step} — closed it to try again",
                    list(self._console),
                )
                self.last_error_at = time.time()
                logger.error("Launch diagnosis: %s", self.last_error)
                await self._hard_reset()
                raise RuntimeError(self.last_error)
            return self._page

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

            launch_kwargs = {
                "headless": True,
                "args": [
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
            self.last_error = launch_diagnosis(str(e), list(self._console))
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
