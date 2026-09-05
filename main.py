import base64
import contextlib
import html
import hmac
import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
import asyncio

import aiohttp
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from typing import Literal, Optional

from browser_service import FEED_QUALITY, FORMAT, QUALITY, BrowserService
from rtm_client import RtmClient
from telemetry_hub import TelemetryHub
from tts_service import generate_speech
from video_feed import FrameBroadcaster, FrameCaptureError

load_dotenv()

# Configurar el logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("http_logger")


async def warmup_browser_when_ready():
    # Hold off while mission gating applies: launching the headless browser
    # renders /sdk, which requires auth, and auth must not run before the
    # user calls /start-mission.
    while os.getenv("MISSION_SLUG") and not auth_response_data:
        await asyncio.sleep(2)
    await browser_service.warmup()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Background task: the page loads /sdk from this same server, which only
    # accepts connections after startup yields — never await warm-up here.
    app.state.http_session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=15)
    )
    warmup_task = asyncio.create_task(warmup_browser_when_ready())
    odo_task = asyncio.create_task(_odo_ticker())   # ★ OKCREAL: dead-reckoning integrator for RETURN TO DOCK
    yield
    odo_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await odo_task
    warmup_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await warmup_task
    cancel_control_watchdog()
    await asyncio.gather(
        *(broadcaster.close() for broadcaster in feed_broadcasters.values())
    )
    await browser_service.close()
    await app.state.http_session.close()


app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRODOBOTS_API_URL = os.getenv(
    "FRODOBOTS_API_URL", "https://frodobots-web-api.onrender.com/api/v1"
)

# How long /v2/* waits for a fresh frame before failing. Kept short on
# purpose: with the warm capture loop a healthy camera answers in tens of
# milliseconds, so this budget only matters as "wait for recovery" during a
# transient capture blip — better a fast 404/503 than a multi-second stall.
V2_FRAME_TIMEOUT_S = float(os.getenv("V2_FRAME_TIMEOUT_S", "2"))


# In-memory storage for the response
auth_response_data = {}
checkpoints_list_data = {}
auth_lock = None
auth_lock_loop = None
INGEST_TOKEN = secrets.token_urlsafe(32)

app.mount("/static", StaticFiles(directory="./static"), name="static")

browser_service = BrowserService()
telemetry_hub = TelemetryHub()


# ★ OKCREAL (Sep 5, 2026): rover microphone. PCM16 mono 16 kHz chunks arrive
#   from the headless page over /ws/audio-ingest and fan out to every
#   GET /audio-feed listener. The page tap is switched on when the first
#   listener connects and off when the last one leaves, so the rover's SIM
#   carries no extra load while nobody is listening (the mic is already in the
#   Agora channel either way; this only adds server → listener traffic).
class AudioHub:
    def __init__(self):
        self._clients: set[asyncio.Queue] = set()
        self.ingest_connected = False
        self.last_chunk_at: Optional[float] = None

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=64)
        self._clients.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        self._clients.discard(q)

    @property
    def listeners(self) -> int:
        return len(self._clients)

    def publish(self, chunk: bytes):
        self.last_chunk_at = time.time()
        for q in list(self._clients):
            if q.full():
                try:
                    q.get_nowait()   # drop the oldest — a slow listener never builds a lag
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(chunk)
            except asyncio.QueueFull:
                pass


audio_hub = AudioHub()
_audio_tap_lock = asyncio.Lock()


async def _audio_tap_sync():
    """Switch the page tap to match whether anyone is listening."""
    async with _audio_tap_lock:
        want = audio_hub.listeners > 0
        try:
            result = await browser_service.audio_tap(want)
            logger.info("audio tap %s: %s", "on" if want else "off", result)
        except Exception as e:
            logger.warning("audio tap %s failed: %s", "on" if want else "off", e)

feed_broadcasters = {
    "front": FrameBroadcaster(browser_service.front_feed),
    "rear": FrameBroadcaster(browser_service.rear_feed),
}


async def external_request(method: str, url: str, **kwargs) -> tuple[int, dict]:
    """Use pooled async HTTP so rover hot paths never block the event loop."""

    async def perform(session: aiohttp.ClientSession):
        debug = os.getenv("DEBUG") == "true"
        if debug:
            logger.info("External %s %s", method.upper(), url)
        async with session.request(method, url, **kwargs) as response:
            try:
                body = await response.json(content_type=None)
            except (aiohttp.ContentTypeError, json.JSONDecodeError):
                body = {"error": await response.text()}

            if debug and response.status >= 400:
                logger.error(
                    "External %s %s failed: %s %s",
                    method.upper(),
                    url,
                    response.status,
                    body,
                )
            return response.status, body

    try:
        session = getattr(app.state, "http_session", None)
        if session and not session.closed:
            return await perform(session)
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15)
        ) as temporary_session:
            return await perform(temporary_session)
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="External API timed out") from exc
    except aiohttp.ClientError as exc:
        raise HTTPException(status_code=502, detail="External API unavailable") from exc


async def get_camera_frame(
    view: str,
) -> tuple[Optional[str], Optional[float]]:
    """Return a shared fresh frame and its capture timestamp."""
    if FORMAT == "jpeg" and QUALITY == FEED_QUALITY:
        broadcaster = feed_broadcasters[view]
        try:
            frame = await broadcaster.get_frame(
                max_age=1 / 30, timeout=V2_FRAME_TIMEOUT_S, fps=30
            )
        except FrameCaptureError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if frame:
            return frame.base64_data, frame.captured_at
        if broadcaster.last_error:
            raise HTTPException(status_code=503, detail=broadcaster.last_error)
        return None, None

    # Preserve explicit png/webp v2 configurations. The default and fastest
    # path is JPEG and shares the feed broadcaster above.
    try:
        packet = await asyncio.wait_for(
            browser_service.configured_frame(view), timeout=V2_FRAME_TIMEOUT_S
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=503, detail=f"{view} camera capture timed out"
        ) from exc
    if packet and packet.get("error"):
        raise HTTPException(status_code=503, detail=packet["error"])
    if not packet or not packet.get("data_url"):
        return None, None
    return packet["data_url"].split(",", 1)[1], float(packet["timestamp"])


async def latest_rover_data() -> dict:
    age = telemetry_hub.age_seconds
    if telemetry_hub.latest is not None and age is not None and age < 5:
        return telemetry_hub.latest
    return await browser_service.data() or {}


@app.get("/feed")
async def feed(view: str = "front", fps: int = 15):
    await need_start_mission()
    if not auth_response_data:
        await auth()

    if view not in feed_broadcasters:
        raise HTTPException(
            status_code=400, detail=f"Invalid view: {view}. Use front or rear"
        )
    fps = max(1, min(fps, 30))
    if view == "rear" and not await browser_service.has_rear_camera():
        raise HTTPException(status_code=404, detail="Rear camera is not available")

    broadcaster = feed_broadcasters[view]
    queue = await broadcaster.subscribe(
        fps, cached_max_age=min(0.5, max(0.1, 2.0 / fps))
    )
    try:
        first_frame = await asyncio.wait_for(queue.get(), timeout=5)
    except asyncio.TimeoutError as exc:
        await broadcaster.unsubscribe(queue)
        detail = f"{view} camera is not ready"
        if broadcaster.last_error:
            detail += f": {broadcaster.last_error}"
        raise HTTPException(status_code=503, detail=detail) from exc
    except asyncio.CancelledError:
        await broadcaster.unsubscribe(queue)
        raise

    async def stream():
        min_interval = 1.0 / fps
        last_sent = 0.0
        try:
            frame = first_frame
            # A None frame is the broadcaster's end-of-stream sentinel
            # (mission ended / server shutting down): finish the response.
            while frame is not None:
                now = time.monotonic()
                if now - last_sent < min_interval * 0.9:
                    frame = await queue.get()
                    continue  # this client asked for fewer fps than we capture
                last_sent = now
                jpeg = frame.jpeg
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                    + jpeg
                    + b"\r\n"
                )
                frame = await queue.get()
        finally:
            await broadcaster.unsubscribe(queue)

    return StreamingResponse(
        stream(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/ws/ingest")
async def ws_ingest(websocket: WebSocket):
    # Private channel for the headless /sdk page; local connections only.
    client_host = websocket.client.host if websocket.client else None
    supplied_token = websocket.query_params.get("token", "")
    if client_host not in ("127.0.0.1", "::1") or not hmac.compare_digest(
        supplied_token, INGEST_TOKEN
    ):
        await websocket.close(code=4403)
        return
    await websocket.accept()
    connection = telemetry_hub.connect_ingest()
    try:
        while True:
            data = await websocket.receive_json()
            telemetry_hub.publish(data)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        telemetry_hub.disconnect_ingest(connection)


@app.websocket("/ws/data")
async def ws_data(websocket: WebSocket):
    await websocket.accept()
    queue = telemetry_hub.subscribe()
    try:
        await websocket.send_json(telemetry_hub.snapshot())
        while True:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=5)
            except asyncio.TimeoutError:
                message = {"type": "status", **telemetry_hub.status()}
            await websocket.send_json(message)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        telemetry_hub.unsubscribe(queue)


@app.get("/status")
async def get_status():
    video = {
        view: {
            "loop_running": broadcaster.loop_running,
            "latest_frame_age_s": broadcaster.latest_age_seconds,
            "captures_total": broadcaster.captures_total,
            "failures_total": broadcaster.failures_total,
            "last_error": broadcaster.last_error,
        }
        for view, broadcaster in feed_broadcasters.items()
    }
    return JSONResponse(
        content={
            "browser_ready": browser_service.is_ready,
            "browser_error": browser_service.last_error,
            "mission_started": bool(auth_response_data)
            or not os.getenv("MISSION_SLUG"),
            "rtm": await browser_service.rtm_health(),
            "video": video,
            **telemetry_hub.status(),
        }
    )


async def auth_common():
    global auth_lock, auth_lock_loop, auth_response_data
    if auth_response_data:
        return auth_response_data

    # Hypercorn's reloader can recreate the application event loop without
    # recreating every imported module object. asyncio locks are loop-bound on
    # Python 3.9, so never carry this coordinator across loop generations.
    running_loop = asyncio.get_running_loop()
    if auth_lock is None or auth_lock_loop is not running_loop:
        auth_lock = asyncio.Lock()
        auth_lock_loop = running_loop

    async with auth_lock:
        if auth_response_data:
            return auth_response_data
        env_tokens = get_env_tokens()
        if env_tokens:
            auth_response_data = env_tokens
            return auth_response_data

        auth_header = os.getenv("SDK_API_TOKEN")
        bot_slug = os.getenv("BOT_SLUG")
        mission_slug = os.getenv("MISSION_SLUG")

        if not auth_header:
            raise HTTPException(
                status_code=500, detail="Authorization header not configured"
            )
        if not bot_slug:
            raise HTTPException(status_code=500, detail="Bot name not configured")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {auth_header}",
        }
        if mission_slug:
            response_data = await start_ride(headers, bot_slug, mission_slug)
        else:
            response_data = await retrieve_tokens(headers, bot_slug)

        auth_response_data = {
            "CHANNEL_NAME": response_data.get("CHANNEL_NAME"),
            "RTC_TOKEN": response_data.get("RTC_TOKEN"),
            "RTM_TOKEN": response_data.get("RTM_TOKEN"),
            "USERID": response_data.get("USERID"),
            "APP_ID": response_data.get("APP_ID"),
            "BOT_UID": response_data.get("BOT_UID"),
            "SPECTATOR_USERID": response_data.get("SPECTATOR_USERID"),
            "SPECTATOR_RTC_TOKEN": response_data.get("SPECTATOR_RTC_TOKEN"),
            "BOT_TYPE": response_data.get("BOT_TYPE", "mini"),
        }
        return auth_response_data


def get_env_tokens():
    channel_name = os.getenv("CHANNEL_NAME")
    rtc_token = os.getenv("RTC_TOKEN")
    rtm_token = os.getenv("RTM_TOKEN")
    userid = os.getenv("USERID")
    app_id = os.getenv("APP_ID")
    bot_uid = os.getenv("BOT_UID")

    if all([channel_name, rtc_token, rtm_token, userid, app_id, bot_uid]):
        return {
            "CHANNEL_NAME": channel_name,
            "RTC_TOKEN": rtc_token,
            "RTM_TOKEN": rtm_token,
            "USERID": userid,
            "APP_ID": app_id,
            "BOT_UID": bot_uid,
            "SPECTATOR_USERID": os.getenv("SPECTATOR_USERID"),
            "SPECTATOR_RTC_TOKEN": os.getenv("SPECTATOR_RTC_TOKEN"),
            "BOT_TYPE": os.getenv("BOT_TYPE", "mini"),
        }
    return None


def _sdk_error_detail(status, fallback):
    """Map known backend failures to safe, user-facing messages. Keyed off the
    HTTP status (not the raw body), so backend internals are never exposed and
    the message stays stable if the backend rewords its errors.
      401 -> the SDK API key didn't resolve to a user
      403 -> the bot is in use / not permitted for SDK
      other -> the caller's generic fallback"""
    if status == 401:
        return "User not found"
    if status == 403:
        return "Bot unavailable for SDK"
    return fallback


async def start_ride(headers, bot_slug, mission_slug):
    start_ride_data = {"bot_slug": bot_slug, "mission_slug": mission_slug}
    status, response_data = await external_request(
        "POST",
        FRODOBOTS_API_URL + "/sdk/start_ride",
        headers=headers,
        json=start_ride_data,
    )
    if status != 200:
        raise HTTPException(
            status_code=status,
            detail=_sdk_error_detail(status, "Bot unavailable for SDK"),
        )
    return response_data


async def end_ride(headers, bot_slug, mission_slug):
    end_ride_data = {"bot_slug": bot_slug, "mission_slug": mission_slug}
    status, response_data = await external_request(
        "POST",
        FRODOBOTS_API_URL + "/sdk/end_ride",
        headers=headers,
        json=end_ride_data,
    )
    if status != 200:
        raise HTTPException(
            status_code=status,
            detail=_sdk_error_detail(status, "Failed to end mission"),
        )
    return response_data


async def retrieve_tokens(headers, bot_slug):
    data = {"bot_slug": bot_slug}
    status, response_data = await external_request(
        "POST", FRODOBOTS_API_URL + "/sdk/token", headers=headers, json=data
    )
    if status != 200:
        raise HTTPException(
            status_code=status,
            detail=_sdk_error_detail(status, "Bot unavailable for SDK"),
        )
    return response_data


async def need_start_mission():
    if not os.getenv("MISSION_SLUG"):
        return
    if auth_response_data:
        return
    raise HTTPException(
        status_code=400, detail="Call /start-mission endpoint to start a mission"
    )


@app.post("/checkpoints-list")
@app.get("/checkpoints-list")
async def checkpoints():
    await need_start_mission()
    await get_checkpoints_list()
    return JSONResponse(content=checkpoints_list_data)


async def get_checkpoints_list():
    global checkpoints_list_data
    auth_header = os.getenv("SDK_API_TOKEN")
    bot_slug = os.getenv("BOT_SLUG")
    mission_slug = os.getenv("MISSION_SLUG")

    if not mission_slug:
        return

    if not auth_header:
        raise HTTPException(
            status_code=500, detail="Authorization header not configured"
        )
    if not bot_slug:
        raise HTTPException(status_code=500, detail="Bot name not configured")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_header}",
    }

    data = {"bot_slug": bot_slug, "mission_slug": mission_slug}

    status, response_data = await external_request(
        "POST",
        FRODOBOTS_API_URL + "/sdk/checkpoints_list",
        headers=headers,
        json=data,
    )
    if status != 200:
        raise HTTPException(
            status_code=status,
            detail="Failed to retrieve checkpoints list",
        )
    checkpoints_list_data = response_data
    return checkpoints_list_data


async def auth():
    await auth_common()
    if not checkpoints_list_data:
        await get_checkpoints_list()
    return JSONResponse(
        content={
            "auth_response_data": auth_response_data,
            "checkpoints_list_data": checkpoints_list_data,
        }
    )


@app.post("/start-mission")
async def start_mission():
    required_env_vars = ["SDK_API_TOKEN", "BOT_SLUG", "MISSION_SLUG"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_vars:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required environment variables: {', '.join(missing_vars)}",
        )

    if not auth_response_data:
        await auth()
    if not checkpoints_list_data:
        await get_checkpoints_list()
    return JSONResponse(
        status_code=200,
        content={
            "message": "Mission started successfully",
            "checkpoints_list": checkpoints_list_data,
        },
    )


@app.post("/end-mission")
async def end_mission():
    required_env_vars = ["SDK_API_TOKEN", "BOT_SLUG", "MISSION_SLUG"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_vars:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required environment variables: {', '.join(missing_vars)}",
        )

    auth_header = os.getenv("SDK_API_TOKEN")
    bot_slug = os.getenv("BOT_SLUG")
    mission_slug = os.getenv("MISSION_SLUG")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_header}",
    }

    # Ending the remote ride destroys the command path. Do not proceed until
    # the rover has positively received zero motion.
    await _require_confirmed_stop("end the mission")

    try:
        await end_ride(headers, bot_slug, mission_slug)
        cancel_control_watchdog()
        # Clear the stored auth and checkpoints data
        global auth_response_data, checkpoints_list_data
        auth_response_data = {}
        checkpoints_list_data = {}
        await asyncio.gather(
            *(broadcaster.close() for broadcaster in feed_broadcasters.values())
        )
        await browser_service.close()
        return JSONResponse(content={"message": "Mission ended successfully"})
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to end mission: {str(e)}")


def render_template(filename: str, template_vars: dict) -> HTMLResponse:
    with open(filename, "r", encoding="utf-8") as file:
        html_content = file.read()

    for key, value in template_vars.items():
        html_content = html_content.replace(f"{{{{ {key} }}}}", str(value))

    return HTMLResponse(content=html_content, status_code=200)


async def render_index_html(is_spectator: bool):
    await need_start_mission()
    if not auth_response_data:
        await auth()

    token_type: Literal["SPECTATOR_", ""] = "SPECTATOR_" if is_spectator else ""

    template_vars = {
        "appid": html.escape(str(auth_response_data.get("APP_ID", "")), quote=True),
        "rtc_token": html.escape(
            str(auth_response_data.get(f"{token_type}RTC_TOKEN", "")), quote=True
        ),
        "rtm_token": html.escape(
            str("" if is_spectator else auth_response_data.get("RTM_TOKEN", "")),
            quote=True,
        ),
        "channel": html.escape(
            str(auth_response_data.get("CHANNEL_NAME", "")), quote=True
        ),
        "uid": html.escape(
            str(auth_response_data.get(f"{token_type}USERID", "")), quote=True
        ),
        "bot_uid": html.escape(str(auth_response_data.get("BOT_UID", "")), quote=True),
        "ingest_token": html.escape(INGEST_TOKEN, quote=True),
        "checkpoints_list": json.dumps(
            checkpoints_list_data.get("checkpoints_list", [])
        ).replace("</", "<\\/"),
        "map_zoom_level": int(os.getenv("MAP_ZOOM_LEVEL", "18")),
    }

    return render_template("index.html", template_vars)


@app.get("/")
async def get_index(request: Request):
    # The dashboard renders even when the mission hasn't started or auth
    # fails — it degrades to "waiting" states instead of a raw JSON error.
    boot_notice = ""
    if not auth_response_data:
        try:
            await need_start_mission()
            await auth()
        except HTTPException as e:
            boot_notice = e.detail if isinstance(e.detail, str) else "SDK not ready"
        except Exception:
            boot_notice = "SDK auth failed - check the credentials in .env"

    tokens = auth_response_data or {}
    dashboard_config = {
        "appid": tokens.get("APP_ID") or "",
        "rtcToken": tokens.get("SPECTATOR_RTC_TOKEN") or "",
        "channel": tokens.get("CHANNEL_NAME") or "",
        "uid": tokens.get("SPECTATOR_USERID") or "",
        "botUid": tokens.get("BOT_UID") or "",
        "checkpointsList": checkpoints_list_data.get("checkpoints_list", []),
        "mapZoomLevel": int(os.getenv("MAP_ZOOM_LEVEL", "18")),
        "botSlug": os.getenv("BOT_SLUG", ""),
        "missionSlug": os.getenv("MISSION_SLUG", ""),
        "missionStarted": bool(auth_response_data) or not os.getenv("MISSION_SLUG"),
        "bootNotice": str(boot_notice).replace("\n", " "),
    }
    template_vars = {
        "dashboard_config": json.dumps(dashboard_config).replace("</", "<\\/")
    }
    return render_template("dashboard.html", template_vars)


@app.get("/sdk")
async def sdk(request: Request):
    return await render_index_html(is_spectator=False)


@app.post("/control-legacy")
async def control_legacy(request: Request):
    await need_start_mission()
    if not auth_response_data:
        await auth()

    body = await request.json()
    command = body.get("command")
    if not command:
        raise HTTPException(status_code=400, detail="Command not provided")

    arm_control_watchdog(command)
    await _dispatch_legacy_control(command)

    return {"message": "Command sent successfully"}


# Dead-man watchdog: the rover keeps executing its last command until a new
# one arrives, so a broken command path after a motion command means a
# runaway bot. The watchdog arms when a motion command is ACCEPTED (before
# dispatch, covering ambiguous delivery) and, once confirmed deliveries are
# stale for CONTROL_WATCHDOG_S, delivers a CONFIRMED stop (peer receipt) —
# retrying and rebuilding the RTM session until the rover confirms it. Failed
# traffic cannot refresh this deadline. CONTROL_WATCHDOG_S=0 disables it.
CONTROL_WATCHDOG_S = float(os.getenv("CONTROL_WATCHDOG_S", "3"))
WATCHDOG_RETRY_DELAY_S = 1.0
WATCHDOG_RESET_EVERY = 3  # rebuild the browser/RTM session every N failures
SAFETY_STOP_CONFIRM_TIMEOUT_S = float(
    os.getenv("SAFETY_STOP_CONFIRM_TIMEOUT_S", "12")
)

_control_watchdog_task: Optional[asyncio.Task] = None
_confirmed_stop_task: Optional[asyncio.Task] = None
_confirmed_stop_generation = 0
_control_dispatch_lock: Optional[asyncio.Lock] = None
_control_dispatch_lock_loop = None


def _command_is_moving(command) -> bool:
    try:
        return bool(
            float(command.get("linear") or 0) or float(command.get("angular") or 0)
        )
    except (TypeError, ValueError, AttributeError):
        return True  # unparseable command: assume motion, err on the safe side


def cancel_control_watchdog():
    global _control_watchdog_task, _confirmed_stop_task
    if _control_watchdog_task and not _control_watchdog_task.done():
        _control_watchdog_task.cancel()
    if _confirmed_stop_task and not _confirmed_stop_task.done():
        _confirmed_stop_task.cancel()
    _control_watchdog_task = None
    _confirmed_stop_task = None


def _get_control_dispatch_lock() -> asyncio.Lock:
    """Return a lock bound to the current application event loop."""
    global _control_dispatch_lock, _control_dispatch_lock_loop
    running_loop = asyncio.get_running_loop()
    if (
        _control_dispatch_lock is None
        or _control_dispatch_lock_loop is not running_loop
    ):
        _control_dispatch_lock = asyncio.Lock()
        _control_dispatch_lock_loop = running_loop
    return _control_dispatch_lock


def _confirmed_stop_pending() -> bool:
    return bool(_confirmed_stop_task and not _confirmed_stop_task.done())


async def _dispatch_browser_control(command):
    """Order local dispatches and reject motion while a safety stop is pending."""
    stop_generation = _confirmed_stop_generation
    stop_pending_at_start = _confirmed_stop_pending()
    async with _get_control_dispatch_lock():
        stop_overtook_dispatch = stop_generation != _confirmed_stop_generation
        if _command_is_moving(command) and (
            stop_pending_at_start
            or _confirmed_stop_pending()
            or stop_overtook_dispatch
        ):
            raise RuntimeError("Motion rejected because a safety stop took priority")
        return await browser_service.send_message(command)


async def _dispatch_legacy_control(command):
    """Keep a slow legacy REST motion ahead of its trailing safety stop."""
    stop_generation = _confirmed_stop_generation
    stop_pending_at_start = _confirmed_stop_pending()
    async with _get_control_dispatch_lock():
        stop_overtook_dispatch = stop_generation != _confirmed_stop_generation
        if _command_is_moving(command) and (
            stop_pending_at_start
            or _confirmed_stop_pending()
            or stop_overtook_dispatch
        ):
            raise RuntimeError("Motion rejected because a safety stop took priority")
        return await asyncio.to_thread(
            RtmClient(auth_response_data).send_message, command
        )


def arm_control_watchdog(command):
    """Start monitoring a drive without letting failed traffic reset its timer.

    The monitor follows Agora's confirmed-delivery timestamp. Healthy streams
    therefore keep it alive, while synchronously or asynchronously failed
    requests cannot postpone the safety deadline.
    """
    if CONTROL_WATCHDOG_S <= 0 or not _command_is_moving(command):
        return
    global _control_watchdog_task
    if _control_watchdog_task and not _control_watchdog_task.done():
        return
    lamp = command.get("lamp") or 0 if isinstance(command, dict) else 0
    _control_watchdog_task = asyncio.create_task(
        _control_watchdog(lamp, time.time())
    )


async def _recent_delivery_delay(armed_at: float) -> Optional[float]:
    """Return how long a recently confirmed control delivery remains fresh."""
    health = await browser_service.rtm_health()
    if not health:
        return None
    try:
        delivered_at = float(health.get("last_delivered_at"))
    except (TypeError, ValueError):
        return None
    if delivered_at < armed_at:
        return None
    age = max(0.0, time.time() - delivered_at)
    remaining = CONTROL_WATCHDOG_S - age
    return remaining if remaining > 0 else None


def _ensure_confirmed_stop(lamp=0) -> asyncio.Task:
    """Return the one shared stop-delivery task for this safety event."""
    global _confirmed_stop_task, _confirmed_stop_generation
    if not _confirmed_stop_task or _confirmed_stop_task.done():
        _confirmed_stop_generation += 1
        _confirmed_stop_task = asyncio.create_task(_deliver_confirmed_stop(lamp))
    return _confirmed_stop_task


async def _deliver_confirmed_stop(lamp) -> bool:
    stop_command = {"linear": 0, "angular": 0, "lamp": lamp}
    attempt = 0
    delay = WATCHDOG_RETRY_DELAY_S
    while auth_response_data:
        attempt += 1
        try:
            async with _get_control_dispatch_lock():
                if await browser_service.send_message_confirmed(stop_command):
                    logger.warning(
                        "Safety stop confirmed by rover (attempt %s)", attempt
                    )
                    return True
                raise RuntimeError("rover did not confirm the stop")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(
                "Safety stop attempt %s failed: %s",
                attempt,
                str(e).split("\n", 1)[0],
            )
            if attempt % WATCHDOG_RESET_EVERY == 0:
                logger.warning("Rebuilding the browser/RTM session to recover")
                async with _get_control_dispatch_lock():
                    with contextlib.suppress(Exception):
                        await browser_service.reset()
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 5.0)
    logger.info("Safety stop abandoned because the mission session was cleared")
    return False


async def _require_confirmed_stop(reason: str, lamp=0):
    """Block destructive lifecycle transitions until the rover confirms zero."""
    task = _ensure_confirmed_stop(lamp)
    try:
        confirmed = await asyncio.wait_for(
            asyncio.shield(task), timeout=SAFETY_STOP_CONFIRM_TIMEOUT_S
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot {reason}: rover has not confirmed the safety stop",
        ) from exc
    if not confirmed:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot {reason}: mission ended before the safety stop was confirmed",
        )


async def _control_watchdog(lamp, armed_at: float):
    delay = CONTROL_WATCHDOG_S
    while True:
        await asyncio.sleep(delay)
        delay = await _recent_delivery_delay(armed_at)
        if delay is None:
            break
    logger.warning(
        "Dead-man watchdog: no confirmed control delivery for %.1fs -"
        " delivering safety stop",
        CONTROL_WATCHDOG_S,
    )
    if not auth_response_data:
        logger.info("Watchdog safety stop skipped: mission session cleared")
        return
    await asyncio.shield(_ensure_confirmed_stop(lamp))


@app.post("/control")
async def control(request: Request):
    await need_start_mission()
    if not auth_response_data:
        await auth()

    body = await request.json()
    command = body.get("command")
    if not command:
        raise HTTPException(status_code=400, detail="Command not provided")

    # ★ OKCREAL (Sep 5, 2026): a manual drive command or an explicit dock stop
    #   aborts self-docking — the human always wins.
    if _dock_active() and (_command_is_moving(command) or body.get("dock") == "stop"):
        await _dock_cancel("manual control")
    if _return_active() and (_command_is_moving(command) or body.get("dock") == "stop"):
        await _return_cancel("manual control")
    _odo_note(command)
    # Arm BEFORE dispatch: if the send times out ambiguously the rover may
    # still have received the motion command — the watchdog must cover it.
    arm_control_watchdog(command)
    try:
        await _dispatch_browser_control(command)
        return {"message": "Command sent successfully"}
    except Exception as e:
        logger.error("Error sending control command: %s", str(e))
        reason = browser_service.last_error or str(e).split("\n", 1)[0]
        detail = "Failed to send control command"
        if reason:
            detail += f": {reason}"
        raise HTTPException(status_code=500, detail=detail) from e


# ★ OKCREAL (Sep 4, 2026): play / stop a live audio stream on the rover
#   speaker (OKCREAL Radio). Body: {"url": "https://.../stream"}.
# ★ OKCREAL (Sep 4, 2026): STANDBY / WAKE. While the SDK page is joined, the
#   rover streams video into the channel around the clock — battery and data
#   for nobody. Standby tears the page down (rover goes quiet, no session);
#   wake rejoins. Connect calls standby when no console has been open for a
#   while and wake when one opens.
@app.post("/standby")
async def standby():
    try:
        await browser_service.close()
        return {"message": "standby", "browser_ready": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"standby failed: {str(e)}") from e


@app.post("/wake")
async def wake():
    try:
        ok = await browser_service.warmup(max_attempts=3)
        return {"message": "awake" if ok else "wake failed", "browser_ready": bool(ok)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"wake failed: {str(e)}") from e


@app.post("/play-live")
async def play_live(request: Request):
    await need_start_mission()
    if not auth_response_data:
        await auth()
    body = await request.json()
    url = body.get("url")
    if not url or not str(url).startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="url required (http/https)")
    try:
        result = await browser_service.play_live(str(url))
        return {"message": "Live audio started", "result": result}
    except Exception as e:
        logger.error("Error in /play-live: %s", str(e))
        raise HTTPException(status_code=500, detail=f"live audio failed: {str(e)}") from e


@app.post("/stop-live")
async def stop_live():
    try:
        result = await browser_service.stop_live()
        return {"message": "Live audio stopped", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"stop failed: {str(e)}") from e


@app.websocket("/ws/audio-ingest")
async def ws_audio_ingest(websocket: WebSocket):
    # Private channel for the headless /sdk page; local connections only.
    client_host = websocket.client.host if websocket.client else None
    supplied_token = websocket.query_params.get("token", "")
    if client_host not in ("127.0.0.1", "::1") or not hmac.compare_digest(
        supplied_token, INGEST_TOKEN
    ):
        await websocket.close(code=4403)
        return
    await websocket.accept()
    audio_hub.ingest_connected = True
    try:
        while True:
            chunk = await websocket.receive_bytes()
            if chunk:
                audio_hub.publish(chunk)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        audio_hub.ingest_connected = False


@app.get("/audio-feed")
async def audio_feed():
    """Rover microphone as a raw PCM stream: signed 16-bit LE, mono, 16 kHz."""
    await need_start_mission()
    if not auth_response_data:
        await auth()
    queue = audio_hub.subscribe()
    asyncio.create_task(_audio_tap_sync())

    async def stream():
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield b"\x00\x00"   # one silent sample keeps the connection alive while the tap is (re)arming (never an empty chunk)
                    continue
                yield chunk
        finally:
            audio_hub.unsubscribe(queue)
            asyncio.create_task(_audio_tap_sync())

    return StreamingResponse(
        stream(),
        media_type="application/octet-stream",
        headers={"X-Audio-Format": "pcm_s16le;rate=16000;channels=1", "Cache-Control": "no-store"},
    )


# ★ OKCREAL (Sep 5, 2026 — Cap: "use the front camera to find the dock, then
#   when close bust a 180, then switch to rear camera and dock"). Two-stage
#   self-dock, v2. The dock stand carries a full-sheet deep-magenta flyer with
#   an ArUco 4x4_50 tag (ID 7). Measured on Cap's console shots today:
#     • the magenta sheet is the far cue (a few px across the room);
#     • the tag decodes from ~2.5 m in at feed resolution;
#     • the REAR camera stream is horizontally MIRRORED (backup-cam style) —
#       the tag only decodes after flipping; the geometry uses the raw frame
#       since raw image-right is the rover's right. Mirroring is auto-detected
#       per camera by which orientation decodes, so a firmware change can't
#       silently break it.
#   Phases: FIND/APPROACH on the front cam driving forward until the tag is
#   DOCK_TAG_TURN_PX wide (~1 m) → TURN 180° on the compass heading (sign
#   learned from the first second of rotation) → BACK on the rear cam:
#   center the tag/sheet (steering sign learned adaptively — if the error
#   grows four steps running, the sign flips), reverse at creep, hold the
#   mat rails for the last stretch → DOCKED when the wheels stall against the
#   stand. Lost marker → stop, slow search turn, give up. Manual drive or
#   ALL STOP aborts. Every tunable is a DOCK_* env var.
import base64  # noqa: E402
import re  # noqa: E402
import math  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

DOCK = {
    "tag_id": int(os.getenv("DOCK_TAG_ID", "7")),
    # mat rails (red-magenta chevrons): hue 146–158 on camera
    "hue_lo": int(os.getenv("DOCK_HUE_LO", "135")), "hue_hi": int(os.getenv("DOCK_HUE_HI", "180")),
    "sat_min": int(os.getenv("DOCK_SAT_MIN", "60")), "val_min": int(os.getenv("DOCK_VAL_MIN", "60")),
    # flyer sheet (deep magenta print reads lavender on camera): hue 126–142, bright
    "sheet_hue_lo": int(os.getenv("DOCK_SHEET_HUE_LO", "126")), "sheet_hue_hi": int(os.getenv("DOCK_SHEET_HUE_HI", "142")),
    "sheet_sat_min": int(os.getenv("DOCK_SHEET_SAT_MIN", "60")), "sheet_val_min": int(os.getenv("DOCK_SHEET_VAL_MIN", "120")),
    "min_ratio": float(os.getenv("DOCK_MIN_RATIO", "0.008")),
    "lane_half": float(os.getenv("DOCK_LANE_HALF", "0.33")),
    "sheet_floor_y": float(os.getenv("DOCK_SHEET_FLOOR_Y", "0.70")),   # small blobs below this line are floor clutter, not the sheet
    "sheet_ceiling_y": float(os.getenv("DOCK_SHEET_CEILING_Y", "0.18")),   # blobs above this line are lights, not the sheet
    # final approach (Cap: "rolling back very slowly until it receives charge - that's when it knows it's arrived")
    "final_z_m": float(os.getenv("DOCK_FINAL_Z_M", "0.32")),        # tag lost inside this range -> blind final roll
    "rev_final": float(os.getenv("DOCK_REV_FINAL", "0.07")),
    "final_max_s": float(os.getenv("DOCK_FINAL_MAX_S", "10")),      # blind roll cap
    "charge_wait_s": float(os.getenv("DOCK_CHARGE_WAIT_S", "240")), # how long to wait for the battery to start rising
    "nudge_s": float(os.getenv("DOCK_NUDGE_S", "0.8")),             # extra push if no charge after the wait
    "yaw_min_px": float(os.getenv("DOCK_YAW_MIN_PX", "60")),        # pose yaw is only trusted when the tag is this big
    # ★ Sep 5 run 3 (console: "APPROACH · FRONT CAM · driving to the dock · sheet 1%" while the dock was
    #   behind the rover): a sheet-only cue must PROVE itself — once it is DOCK_SHEET_CONFIRM wide the tag
    #   should decode; if it doesn't within DOCK_SHEET_PROVE_S the blob is a purple light, not the dock,
    #   and that bearing is ignored for DOCK_BLACKLIST_S.
    "sheet_confirm": float(os.getenv("DOCK_SHEET_CONFIRM", "0.06")),
    "sheet_prove_s": float(os.getenv("DOCK_SHEET_PROVE_S", "2.5")),
    "blacklist_s": float(os.getenv("DOCK_BLACKLIST_S", "45")),
    "fwd": float(os.getenv("DOCK_FWD", "0.16")), "fwd_near": float(os.getenv("DOCK_FWD_NEAR", "0.10")),
    "rev": float(os.getenv("DOCK_REV", "0.12")), "rev_near": float(os.getenv("DOCK_REV_NEAR", "0.08")),
    "tag_turn_px": float(os.getenv("DOCK_TAG_TURN_PX", "90")),    # fallback (no pose): tag side px @640 when it's time to turn
    "tag_m": float(os.getenv("DOCK_TAG_M", "0.1524")),              # printed tag side: 6 in
    "hfov_deg": float(os.getenv("DOCK_HFOV_DEG", "110")),           # camera horizontal FOV estimate (Mini+ is wide)
    "stage_m": float(os.getenv("DOCK_STAGE_M", "0.6")),             # ★ Cap: turn around ~2 ft from the stand — use the sharp front camera all the way in
    "stage_tol_m": float(os.getenv("DOCK_STAGE_TOL_M", "0.2")),
    "yaw_ok_deg": float(os.getenv("DOCK_YAW_OK_DEG", "18")),        # on-axis enough to turn around
    "sheet_stop_ratio": float(os.getenv("DOCK_SHEET_STOP_RATIO", "0.32")),  # hard stop: sheet this wide = we are at the stand
    "tag_near_px": float(os.getenv("DOCK_TAG_NEAR_PX", "70")),
    "turn_gain": float(os.getenv("DOCK_TURN_GAIN", "1.4")), "turn_max": float(os.getenv("DOCK_TURN_MAX", "0.42")),
    "center_tol": float(os.getenv("DOCK_CENTER_TOL", "0.05")), "turn_only": float(os.getenv("DOCK_TURN_ONLY", "0.18")),
    "spin": float(os.getenv("DOCK_SPIN", "0.70")),               # ★ run 5 log: 0.50 managed ~4°/pulse — the 180 timed out at 97°
    "ang_min_inplace": float(os.getenv("DOCK_ANG_MIN_INPLACE", "0.38")),  # ★ Sep 5 run 2: 0.22 didn't turn the rover — motor dead zone
    "ang_min_moving": float(os.getenv("DOCK_ANG_MIN_MOVING", "0.15")),
    "spin_tol_deg": float(os.getenv("DOCK_SPIN_TOL_DEG", "8")), "spin_timeout_s": float(os.getenv("DOCK_SPIN_TIMEOUT_S", "90")),   # ceiling only; the turn is judged in degrees
    "spin_blind_s": float(os.getenv("DOCK_SPIN_BLIND_S", "3.5")),   # no heading data: timed turn
    # ★ Sep 5 run 4 flight log: at 0.40 the search turned ~2°/s (25 s covered ~60°) — the dock was never
    #   swept into view. Search is now spin-and-look by DEGREES: strong pulses until ≥380° of compass
    #   has been covered (or DOCK_SEARCH_S as the ceiling), checking BOTH cameras each stop.
    "search_turn": float(os.getenv("DOCK_SEARCH_TURN", "0.65")), "search_s": float(os.getenv("DOCK_SEARCH_S", "90")),
    "search_pulse_s": float(os.getenv("DOCK_SEARCH_PULSE_S", "0.8")),
    "timeout_s": float(os.getenv("DOCK_TIMEOUT_S", "240")), "stall_s": float(os.getenv("DOCK_STALL_S", "2.0")),
    "hz": float(os.getenv("DOCK_HZ", "5")),
    # ★ Sep 5 (Cap: "it's trying to find the image while moving which causes blur"):
    #   turns and searches PULSE — rotate, stop, settle, look — and a lost marker
    #   is only declared lost from a settled (stopped) frame.
    "pulse_s": float(os.getenv("DOCK_PULSE_S", "0.5")),          # rotate this long per pulse
    "settle_s": float(os.getenv("DOCK_SETTLE_S", "0.45")),        # stop this long before looking
    "front_sign": float(os.getenv("DOCK_FRONT_SIGN", "-1")),   # angular = sign * x_err * gain  (front cam, forward)
    "rear_sign": float(os.getenv("DOCK_REAR_SIGN", "1")),      # rear cam (raw/mirrored frame), reversing
}
# ★ Sep 5, 2026 (Cap: "create a diagnostic download so you can see the events").
#   Every run keeps a flight log: phase changes, every control tick (what each
#   camera saw, the command sent, heading, wheel state), the learned mirror/sign
#   values, the tunables in force, and small JPEG snapshots at each phase change.
#   GET /dock/log returns the last run as JSON (Connect proxies it as a download).
_docklog = {"run_id": None, "started_at": None, "events": [], "ticks": [], "snaps": [], "config": None}
_DOCKLOG_MAX_TICKS = 3000
_DOCKLOG_MAX_SNAPS = 16
def _docklog_reset():
    _docklog.update({"run_id": time.strftime("%Y%m%d-%H%M%S"), "started_at": time.time(), "events": [], "ticks": [], "snaps": [], "config": dict(DOCK)})
def _docklog_event(kind, text, extra=None):
    e = {"t": round(time.time() - (_docklog["started_at"] or time.time()), 2), "kind": kind, "text": text}
    if extra: e.update(extra)
    _docklog["events"].append(e)
def _docklog_tick(stage, cam, see, linear, angular, note=None):
    if len(_docklog["ticks"]) >= _DOCKLOG_MAX_TICKS:
        return
    tag = see and see.get("tag"); sheet = see and see.get("sheet"); d = telemetry_hub.latest or {}
    _docklog["ticks"].append({
        "t": round(time.time() - (_docklog["started_at"] or time.time()), 2), "stage": stage, "cam": cam,
        "cmd": [round(linear, 3), round(angular, 3)],
        "tag": ({k: tag.get(k) for k in ("x_err", "side_px", "x_m", "z_m", "yaw_deg", "stage_bearing_deg", "stage_dist_m", "mirrored")} if tag else None),
        "sheet": ({k: sheet.get(k) for k in ("x_err", "ratio", "cy")} if sheet else None),
        "lane_err": see.get("lane_err") if see else None,
        "hdg": _dock_heading(), "rpms0": _dock_rpms_zero(), "bat": d.get("battery"), "spd": d.get("speed"), "note": note,
    })
def _docklog_snap(label, jpeg):
    if not jpeg or len(_docklog["snaps"]) >= _DOCKLOG_MAX_SNAPS:
        return
    try:
        img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        h, w = img.shape[:2]; sm = cv2.resize(img, (320, int(h * 320.0 / w)))
        ok, enc = cv2.imencode(".jpg", sm, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if ok:
            _docklog["snaps"].append({"t": round(time.time() - (_docklog["started_at"] or time.time()), 2), "label": label, "jpeg_b64": base64.b64encode(enc.tobytes()).decode()})
    except Exception:
        pass

_dock_blacklist = {"front": [], "rear": []}   # cam → [(x_err, until)]
def _dock_blacklisted(cam, x_err):
    now = time.time(); lst = [b for b in _dock_blacklist.get(cam, []) if b[1] > now]; _dock_blacklist[cam] = lst
    return any(abs(x_err - b[0]) < 0.12 for b in lst)
_dock = {"task": None, "state": "idle", "phase": None, "since": None, "reason": None, "sense": None,
         "started_at": None, "last_seen": None, "cmds": 0, "cam": None,
         "mirror": {"front": None, "rear": None}, "sign": {"front": DOCK["front_sign"], "rear": DOCK["rear_sign"], "spin": 1.0}}

_aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
_aruco_params = cv2.aruco.DetectorParameters()
_aruco_params.adaptiveThreshWinSizeMin = 3; _aruco_params.adaptiveThreshWinSizeMax = 35; _aruco_params.adaptiveThreshWinSizeStep = 4
_aruco_params.minMarkerPerimeterRate = 0.01
_aruco = cv2.aruco.ArucoDetector(_aruco_dict, _aruco_params)


def dock_sense(jpeg):
    img=cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR)
    if img is None: return None
    h,w=img.shape[:2]
    if w>480:
        img=cv2.resize(img,(480,int(h*480.0/w))); h,w=img.shape[:2]
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,(DOCK["hue_lo"],DOCK["sat_min"],DOCK["val_min"]),(DOCK["hue_hi"],255,255))
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    cs,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    blobs=[]
    for c in cs:
        a=cv2.contourArea(c)
        if a<6: continue
        x,y,bw,bh=cv2.boundingRect(c)
        blobs.append({"x":(x+bw/2)/w,"y":(y+bh/2)/h,"w":bw/w,"h":bh/h,"asp":bw/float(bh),"a":a/(w*h)})
    if not blobs: return {"seen":False}
    # the bullseye up close: a big round blob anywhere in the frame = we are at the stand
    for b in sorted(blobs,key=lambda b:-b["a"]):
        r=b["w"]; asp=b["asp"]
        if r>=DOCK["dot_done_ratio"] and 0.7<asp<1.5:
            return {"seen":True,"dot":{"x_err":b["x"]-0.5,"ratio":round(r,4),"cy":round(b["y"],3),"big":True},"rails":None,"lane_err":None}
    # rails: blobs in the lower 60% of the frame; dot: the highest sizeable blob above them
    rails=[b for b in blobs if b["y"]>0.35]
    top=[b for b in blobs if b["y"]<=0.35 and b["w"]>=DOCK["min_ratio"]]
    dot=None
    if top:
        d=max(top,key=lambda b:b["a"]); dot={"x_err":d["x"]-0.5,"ratio":round(d["w"],4),"cy":round(d["y"],3)}
    out={"seen":True,"dot":dot,"rails":None,"lane_err":None}
    if rails:
        xs=sorted(b["x"] for b in rails)
        med=np.median(xs)
        left=[b for b in rails if b["x"]<=med]; right=[b for b in rails if b["x"]>med]
        lx=np.average([b["x"] for b in left],weights=[b["a"] for b in left]) if left else None
        rx=np.average([b["x"] for b in right],weights=[b["a"] for b in right]) if right else None
        if lx is not None and rx is not None and (rx-lx)>0.12:
            center=(lx+rx)/2; kind="both"
        else:
            # one rail (or the two clusters are really one): assume left rail if it sits left of center
            x=np.average([b["x"] for b in rails],weights=[b["a"] for b in rails])
            center = x+DOCK["lane_half"] if x<0.5 else x-DOCK["lane_half"]; kind="left" if x<0.5 else "right"
        area=sum(b["a"] for b in rails)
        out["rails"]={"kind":kind,"left":(round(float(lx),3) if lx is not None else None),"right":(round(float(rx),3) if rx is not None else None),"area":round(float(area),4),"n":len(rails)}
        out["lane_err"]=round(float(center)-0.5,3)
    return out



def _dock_find_tag(gray, w):
    """Tag corners → bearing, size, and an approximate 3-D pose (metres, camera
    frame: x right, y down, z forward). Intrinsics are estimated from the
    camera's horizontal field of view (DOCK_HFOV_DEG) — good enough for a
    staging point, and the sign of the lateral offset / yaw is what matters."""
    corners, ids, _ = _aruco.detectMarkers(gray)
    if ids is None:
        return None
    h = gray.shape[0]
    for c, i in zip(corners, ids.ravel()):
        if int(i) != DOCK["tag_id"]:
            continue
        q = c[0]
        side = float(max(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[1] - q[2])))
        out = {"cx": float(q[:, 0].mean()) / w, "cy": float(q[:, 1].mean()) / h, "side_px": round(side * (640.0 / w), 1)}
        try:
            fx = (w / 2.0) / math.tan(math.radians(DOCK["hfov_deg"]) / 2.0)
            K = np.array([[fx, 0, w / 2.0], [0, fx, h / 2.0], [0, 0, 1]], dtype=np.float64)
            L = DOCK["tag_m"] / 2.0
            obj = np.array([[-L, L, 0], [L, L, 0], [L, -L, 0], [-L, -L, 0]], dtype=np.float64)
            ok, rvec, tvec = cv2.solvePnP(obj, q.astype(np.float64), K, np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                R, _ = cv2.Rodrigues(rvec)
                t = tvec.ravel()
                n = R[:, 2]                        # tag normal in camera coords
                if float(np.dot(n, t)) > 0:        # make it point back toward the camera
                    n = -n
                out["x_m"] = round(float(t[0]), 3); out["z_m"] = round(float(t[2]), 3)
                out["dist_m"] = round(float(np.linalg.norm(t)), 3)
                # yaw of the dock relative to our line of sight: 0 = we are on its axis
                out["yaw_deg"] = round(math.degrees(math.atan2(float(n[0]), float(-n[2]))), 1)
                # staging point: DOCK_STAGE_M out along the dock's axis, in camera coords
                P = t + n * DOCK["stage_m"]
                out["stage_x_m"] = round(float(P[0]), 3); out["stage_z_m"] = round(float(P[2]), 3)
                out["stage_bearing_deg"] = round(math.degrees(math.atan2(float(P[0]), float(P[2]))), 1)
                out["stage_dist_m"] = round(float(math.hypot(P[0], P[2])), 3)
        except Exception as e:
            out["pose_error"] = str(e)[:60]
        return out
    return None


def dock_see(jpeg: bytes, cam: str):
    """Everything the docker can see in one frame: tag (with mirror auto-detect),
    magenta sheet, mat rails. x_err values are in RAW frame coordinates."""
    img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    out = {"cam": cam, "tag": None, "sheet": None, "rails": None, "lane_err": None}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mir = _dock["mirror"].get(cam)
    order = [False, True] if mir is None else [bool(mir), not bool(mir)]
    for flipped in order:
        t = _dock_find_tag(cv2.flip(gray, 1) if flipped else gray, w)
        if t:
            if flipped:
                # geometry back into RAW-frame terms (raw image-right = rover's right on a mirrored feed)
                t["cx"] = 1.0 - t["cx"]
                for k in ("x_m", "stage_x_m"):
                    if k in t: t[k] = round(-t[k], 3)
                if "yaw_deg" in t: t["yaw_deg"] = round(-t["yaw_deg"], 1)
                if "stage_bearing_deg" in t: t["stage_bearing_deg"] = round(-t["stage_bearing_deg"], 1)
            t["x_err"] = round(t["cx"] - 0.5, 3); t["mirrored"] = flipped
            if _dock["mirror"].get(cam) is None:
                _dock["mirror"][cam] = flipped; logger.info("dock: %s camera is %s", cam, "MIRRORED" if flipped else "not mirrored")
                _docklog_event("mirror", "%s camera is %s" % (cam, "MIRRORED" if flipped else "not mirrored"))
            out["tag"] = t
            break
    # magenta sheet: purple-magenta blobs above the floor line
    # (tuned on Cap's Sep 5 shots: the printed sheet reads hue 127–137 and bright;
    #  dark purple shadows, the red-X floor sheet (hue ~165) and the rug are rejected
    #  by hue band, brightness, aspect, and the floor line)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (DOCK["sheet_hue_lo"], DOCK["sheet_sat_min"], DOCK["sheet_val_min"]), (DOCK["sheet_hue_hi"], 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cs:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 4 or bh < 4:
            continue
        asp = bw / float(bh)
        if asp < 0.4 or asp > 2.5:
            continue
        cy = (y + bh / 2.0) / h
        if cy > DOCK["sheet_floor_y"] and bw / float(w) < 0.3:
            continue
        # Sep 5 flight log: purple LED strip lighting along the ceiling read as "sheet 37%"
        # and sent the rover backing toward a lamp. The stand's sheet never sits in the top
        # band of the frame, and once it's more than a few px wide it has the tag's black
        # square inside it - a light strip has no dark interior.
        if cy < DOCK["sheet_ceiling_y"] and bw / float(w) < 0.5:
            continue
        roi = hsv[y:y + bh, x:x + bw]
        if (roi[..., 2] >= DOCK["sheet_val_min"]).mean() < 0.5 or np.median(roi[..., 1]) < DOCK["sheet_sat_min"]:
            continue
        if bw >= 14:
            dark = (roi[..., 2] < 90).mean()
            if dark < 0.05 or dark > 0.65:
                continue
        if _dock_blacklisted(cam, (x + bw / 2.0) / w - 0.5):
            continue
        if best is None or bw * bh > best[2] * best[3]:
            best = (x, y, bw, bh)
    if best:
        x, y, bw, bh = best
        out["sheet"] = {"x_err": round((x + bw / 2.0) / w - 0.5, 3), "ratio": round(bw / float(w), 4), "cy": round((y + bh / 2.0) / h, 3)}
    # mat rails (for the last stretch)
    try:
        r = dock_sense(jpeg)
        if r and r.get("rails"):
            out["rails"] = r["rails"]; out["lane_err"] = r["lane_err"]
    except Exception:
        pass
    return out


def _dock_rpms_zero() -> bool:
    d = telemetry_hub.latest or {}
    rp = d.get("rpms") or []
    if not rp:
        return False
    try:
        last = rp[-1]
        return all(abs(float(v)) < 1.0 for v in last[:4])
    except Exception:
        return False


def _dock_heading():
    d = telemetry_hub.latest or {}
    try:
        o = d.get("orientation")
        return None if o is None else float(o) % 360.0
    except Exception:
        return None


async def _dock_send(linear: float, angular: float):
    # motor dead zone: a small angular command doesn't move the rover at all
    if angular:
        floor = DOCK["ang_min_inplace"] if not linear else DOCK["ang_min_moving"]
        if abs(angular) < floor:
            angular = floor if angular > 0 else -floor
    cmd = {"linear": round(linear, 3), "angular": round(angular, 3), "lamp": 0}
    arm_control_watchdog(cmd)
    _dock["cmds"] += 1
    _dock["_last_cmd"] = (cmd["linear"], cmd["angular"])
    _odo_note(cmd)
    try:
        await _dispatch_browser_control(cmd)
    except Exception as e:
        logger.warning("dock: control send failed: %s", e)


def _dock_set(phase, reason=None):
    if _dock["phase"] != phase:
        _dock["phase"] = phase; _dock["since"] = time.time()
        logger.info("dock: %s%s", phase, (" — " + reason) if reason else "")
        _docklog_event("phase", phase + ((" — " + reason) if reason else ""), {"cam": _dock.get("cam"), "state": _dock.get("state")})
        _dock["_snap_wanted"] = phase
    if reason:
        _dock["reason"] = reason


async def _dock_frame(cam: str):
    b = feed_broadcasters[cam]
    try:
        fr = await b.get_frame(max_age=0.4, timeout=2.0, fps=int(DOCK["hz"]))
    except Exception as e:
        logger.warning("dock: %s frame error: %s", cam, e)
        _docklog_event("frame_error", cam + ": " + str(e)[:100]); return None
    if fr is None:
        _docklog_event("frame_error", cam + ": no frame")
    _dock["_last_frame"] = (cam, fr.jpeg if fr else None)
    return fr


async def _dock_settled_look(cam: str):
    """Stop, let the camera settle, take a FRESH frame, and sense it."""
    await _dock_send(0, 0)
    await asyncio.sleep(DOCK["settle_s"])
    b = feed_broadcasters[cam]
    try:
        fr = await b.get_frame(max_age=0.15, timeout=2.0, fps=int(DOCK["hz"]))
    except Exception as e:
        _docklog_event("frame_error", cam + " (settled): " + str(e)[:100]); fr = None
    _dock["_last_frame"] = (cam, fr.jpeg if fr else None)
    see = dock_see(fr.jpeg, cam) if fr else None
    _dock["sense"] = see; _dock["cam"] = cam
    return see


def _dock_log_tick(stage, note=None):
    """Called once per control-loop iteration by the stages (after the send)."""
    cam, jpeg = _dock.get("_last_frame", (None, None))
    lin, ang = _dock.get("_last_cmd", (0, 0))
    _docklog_tick(stage, cam, _dock.get("sense"), lin, ang, note)
    want = _dock.pop("_snap_wanted", None)
    if want and jpeg:
        _docklog_snap(want + " (" + str(cam) + ")", jpeg)


async def _dock_search(primary_cam: str):
    """Spin-and-look until either camera has the dock. Returns ('front'|'rear', see) or None.
    Rotation is measured on the compass so a slow rover still completes the sweep."""
    h0 = _dock_heading(); hp = h0; turned = 0.0; t0 = time.time()
    cams = [primary_cam] + (["rear" if primary_cam == "front" else "front"] if await browser_service.has_rear_camera() else [])
    _dock_set("search", "sweeping for the dock (both cameras)")
    while time.time() - t0 < DOCK["search_s"] and turned < 380:
        await _dock_send(0, DOCK["search_turn"])
        await asyncio.sleep(DOCK["search_pulse_s"])
        for cam in cams:
            see = await _dock_settled_look(cam)
            tag = see and see.get("tag"); sheet = see and see.get("sheet")
            if tag or (sheet and sheet["ratio"] >= DOCK["min_ratio"] and not _dock_blacklisted(cam, sheet["x_err"])):
                _docklog_event("search_hit", "%s camera has the dock after ~%d° (%s)" % (cam, turned, "tag" if tag else "sheet %d%%" % int(sheet["ratio"] * 100)))
                return (cam, see)
        h = _dock_heading()
        if h is not None and hp is not None:
            turned += abs(((h - hp + 540.0) % 360.0) - 180.0); hp = h
        _dock_set("search", "sweeping · ~%d° covered · %ds" % (turned, time.time() - t0))
        _dock_log_tick("search", "turned~%d" % turned)
    await _dock_send(0, 0)
    _docklog_event("search_miss", "full sweep (~%d°) with no dock in either camera" % turned)
    return None


class _SignLearner:
    """Flip the steering sign if the error keeps growing under correction."""
    def __init__(self, cam):
        self.cam = cam; self.prev = None; self.bad = 0
    def sign(self):
        return _dock["sign"][self.cam]
    def observe(self, err, steered):
        if self.prev is not None and steered:
            if abs(err) > abs(self.prev) + 0.01:
                self.bad += 1
            else:
                self.bad = 0
            if self.bad >= 4:
                _dock["sign"][self.cam] *= -1; self.bad = 0
                logger.warning("dock: %s steering sign flipped to %+d (error kept growing)", self.cam, int(_dock["sign"][self.cam]))
                _docklog_event("sign_flip", "%s steering sign → %+d" % (self.cam, int(_dock["sign"][self.cam])))
        self.prev = err


async def _dock_stage_approach():
    """Front cam, forward. Not "drive at the tag": drive to a STAGING POINT on the
    dock's axis (DOCK_STAGE_M out from the stand), then face the dock squarely.
    Returns True when it's time to turn around. Hard stops so it can never
    plough into the stand: sheet filling the frame, tag closer than the
    staging distance, or the tag too big when no pose is available."""
    learner = _SignLearner("front"); search_dir = 1.0; faced_since = None; sheet_big_since = None
    while True:
        now = time.time()
        if now - _dock["started_at"] > DOCK["timeout_s"]:
            await _dock_send(0, 0); _dock["state"] = "failed"; _dock_set("timeout", "gave up after %ds" % DOCK["timeout_s"]); return False
        fr = await _dock_frame("front"); see = dock_see(fr.jpeg, "front") if fr else None
        _dock["sense"] = see; _dock["cam"] = "front"
        tag = see and see.get("tag"); sheet = see and see.get("sheet")
        # ---- hard stops ----
        if sheet and sheet["ratio"] >= DOCK["sheet_stop_ratio"]:
            await _dock_send(0, 0); _dock_set("staged", "at the stand (sheet fills the frame) — turning"); return True
        if tag and tag.get("z_m") is not None and tag["z_m"] < DOCK["stage_m"] * 0.75 and abs(tag["x_err"]) < 0.2:
            await _dock_send(0, 0); _dock_set("staged", "inside staging distance (%.2f m) — turning" % tag["z_m"]); return True
        if tag and tag.get("z_m") is None and tag["side_px"] >= DOCK["tag_turn_px"]:
            await _dock_send(0, 0); _dock_set("staged", "tag %dpx (no pose) — turning" % tag["side_px"]); return True
        # a sheet that is big enough to carry a readable tag but never decodes is a light, not the dock
        if tag:
            sheet_big_since = None
        elif sheet and sheet["ratio"] >= DOCK["sheet_confirm"]:
            sheet_big_since = sheet_big_since or now
            if now - sheet_big_since > DOCK["sheet_prove_s"]:
                _dock_blacklist["front"].append((sheet["x_err"], now + DOCK["blacklist_s"]))
                _docklog_event("false_sheet", "front: sheet %d%% at %+d%% never showed a tag — ignoring that bearing" % (int(sheet["ratio"] * 100), int(sheet["x_err"] * 100)))
                sheet = None; sheet_big_since = None
        else:
            sheet_big_since = None
        tgt = tag or sheet
        if not tgt:
            # a moving frame said nothing — stop and look before believing it
            see = await _dock_settled_look("front"); tag = see and see.get("tag"); sheet = see and see.get("sheet") if not (see and see.get("sheet") and _dock_blacklisted("front", see["sheet"]["x_err"])) else None; tgt = tag or sheet
        if not tgt:
            # look BEHIND first: if the rear camera already has the dock we are facing away — skip the approach and turn
            try:
                if await browser_service.has_rear_camera():
                    rs = await _dock_settled_look("rear"); rt = rs and rs.get("tag"); rsh = rs and rs.get("sheet")
                    _docklog_event("rear_look", "rear camera: %s" % ("TAG" if rt else ("sheet %d%%" % int(rsh["ratio"] * 100) if rsh else "nothing")))
                    if rt or (rsh and rsh["ratio"] >= DOCK["sheet_confirm"] and not _dock_blacklisted("rear", rsh["x_err"])):
                        _dock["cam"] = "rear"; _dock_set("rear_first", "dock is behind us already — skipping the approach and turn"); return "back"
                    _dock["cam"] = "front"
            except Exception:
                pass
        if not tgt:
            _dock_set("lost", "dock not in view — sweeping")
            hit = await _dock_search("front")
            if not hit:
                await _dock_send(0, 0); _dock["state"] = "failed"; _dock_set("no_target", "swept a full circle and never saw the dock"); return False
            cam, see = hit
            if cam == "rear":
                _dock["cam"] = "rear"; _dock_set("rear_first", "dock found behind us — skipping the approach and turn"); return "back"
            faced_since = None
            continue
        _dock["last_seen"] = now
        # ---- with a pose: navigate to the staging point, then square up ----
        if tag and tag.get("stage_dist_m") is not None:
            sd = tag["stage_dist_m"]; sb = tag["stage_bearing_deg"]; yaw = tag.get("yaw_deg", 0.0)
            # ★ run 5 log: with the staging point only ~0.4 m away its bearing swung ±80° every tick and the
            #   rover spun in place hunting it, losing the tag three times. Close to the point, the bearing is
            #   meaningless — what matters is being about DOCK_STAGE_M from the tag, facing it, near its axis.
            z = tag.get("z_m") or 0.0
            near_band = abs(z - DOCK["stage_m"]) <= 0.25 and abs(tag["x_err"]) < 0.22 and abs(yaw) <= DOCK["yaw_ok_deg"] * 1.5
            if sd > DOCK["stage_tol_m"] and not (sd < 0.6 and abs(sb) > 45) and not near_band:
                # bearing to the staging point drives the steering; forward when roughly pointed at it
                err = max(-0.5, min(0.5, sb / 60.0))
                angular = max(-DOCK["turn_max"], min(DOCK["turn_max"], learner.sign() * err * DOCK["turn_gain"] * 1.6))
                linear = 0.0 if abs(sb) > 35 else (DOCK["fwd_near"] if sd < 0.6 else DOCK["fwd"])
                _dock_set("stage", "to staging point · %.2f m, %+d° · dock yaw %+d°" % (sd, sb, yaw))
                learner.observe(err, angular != 0.0); faced_since = None
                await _dock_send(linear, angular)
            else:
                # at the staging point: rotate in place until the dock is centered and we're on its axis
                x_err = tag["x_err"]
                if abs(x_err) < DOCK["center_tol"] and abs(yaw) <= DOCK["yaw_ok_deg"]:
                    faced_since = faced_since or now
                    await _dock_send(0, 0)
                    if now - faced_since > 0.6:
                        _dock_set("staged", "on the dock axis (yaw %+d°) — turning" % yaw); return True
                else:
                    faced_since = None
                    angular = max(-DOCK["turn_max"], min(DOCK["turn_max"], learner.sign() * x_err * DOCK["turn_gain"]))
                    if abs(x_err) < DOCK["center_tol"] and abs(yaw) > DOCK["yaw_ok_deg"]:
                        # centered but off-axis: back off a little on an arc to get onto the axis
                        angular = DOCK["turn_max"] * 0.6 * (1 if yaw > 0 else -1) * learner.sign()
                        await _dock_send(-DOCK["fwd_near"], angular); _dock_set("square", "squaring up · dock yaw %+d°" % yaw)
                    else:
                        await _dock_send(0, angular); _dock_set("face", "facing the dock · %+d%%" % int(x_err * 100))
                    learner.observe(x_err, True)
            _dock_log_tick("approach")
            await asyncio.sleep(1.0 / DOCK["hz"]); continue
        # ---- no pose (sheet only, or tag without pose): bearing-only approach ----
        x_err = tgt["x_err"]; search_dir = 1.0 if x_err < 0 else -1.0
        angular = max(-DOCK["turn_max"], min(DOCK["turn_max"], learner.sign() * x_err * DOCK["turn_gain"]))
        if abs(x_err) > DOCK["turn_only"]:
            linear = 0.0; _dock_set("align", "centering the dock (front)")
        else:
            if abs(x_err) < DOCK["center_tol"]:
                angular = 0.0
            linear = DOCK["fwd_near"] if (tag and tag["side_px"] >= DOCK["tag_near_px"]) else DOCK["fwd"]
            _dock_set("approach", "driving to the dock" + (" · tag %dpx" % tag["side_px"] if tag else " · sheet %d%%" % int(sheet["ratio"] * 100)))
        learner.observe(x_err, angular != 0.0)
        await _dock_send(linear, angular)
        await asyncio.sleep(1.0 / DOCK["hz"])


async def _dock_stage_turn():
    """About-face. ★ Sep 5 run 2: a compass-only 180 left the rover still facing
    the dock (rear cam blind, 2 minutes of searching). The turn is now ended by
    VISION: spin until the rear camera sees the tag or sheet near center; the
    compass only reports progress. Up to ~400° of rotation before giving up."""
    h0 = _dock_heading(); t0 = time.time(); turned = 0.0; hp = h0
    sign = _dock["sign"]["spin"]
    _dock_set("turn", "180° — spinning until the rear camera sees the dock")
    while True:
        if time.time() - t0 > DOCK["spin_timeout_s"]:
            await _dock_send(0, 0); _dock["state"] = "failed"; _dock_set("turn_timeout", "spun for %ds without the rear camera finding the dock" % DOCK["spin_timeout_s"]); return False
        await _dock_send(0, DOCK["spin"] * sign)
        await asyncio.sleep(DOCK["pulse_s"])
        see = await _dock_settled_look("rear")          # stop · settle · look (no motion blur)
        h = _dock_heading()
        if h is not None and hp is not None:
            d = ((h - hp + 540.0) % 360.0) - 180.0; turned += abs(d); hp = h
        tag = see and see.get("tag"); sheet = see and see.get("sheet")
        hit = (tag and abs(tag["x_err"]) < 0.22) or (sheet and abs(sheet["x_err"]) < 0.18 and sheet["ratio"] >= 0.03)
        if tag and not hit:
            # ★ run 5 log: the rear camera HAD the tag at +46%…+28% off-center for the last 6 s of the turn
            #   and the turn still timed out. Once the tag is in the rear frame, finish by centering it.
            _dock_set("turn", "180° — rear cam has the tag at %+d%%, centering" % int(tag["x_err"] * 100))
            for _ in range(12):
                await _dock_send(0, DOCK["spin"] * _dock["sign"]["rear"] * (1 if tag["x_err"] > 0 else -1))
                await asyncio.sleep(0.35)
                see = await _dock_settled_look("rear"); t2 = see and see.get("tag")
                if not t2:
                    break
                if abs(t2["x_err"]) > abs(tag["x_err"]) + 0.03:
                    _dock["sign"]["rear"] *= -1; _docklog_event("sign_flip", "rear centering sign → %+d" % int(_dock["sign"]["rear"]))
                tag = t2
                if abs(tag["x_err"]) < 0.22:
                    hit = True; break
        _dock_set("turn", "180° — turned ~%d°, rear cam: %s" % (turned, "TAG" if tag else "sheet" if sheet else "nothing yet"))
        _dock_log_tick("turn", "turned~%d" % turned)
        if hit:
            await _dock_send(0, 0); _dock_set("turned", "rear camera has the dock (~%d° turned)" % turned); return True
        if turned > 400:
            await _dock_send(0, 0); _dock["state"] = "failed"; _dock_set("turn_lost", "full circle and the rear camera never saw the dock"); return False


async def _dock_stage_back():
    """Rear cam (mirrored), reversing onto the mat until the wheels stall."""
    learner = _SignLearner("rear"); search_dir = 1.0; rev_since = None; last_close = None
    while True:
        now = time.time()
        if now - _dock["started_at"] > DOCK["timeout_s"]:
            await _dock_send(0, 0); _dock["state"] = "failed"; _dock_set("timeout", "gave up after %ds" % DOCK["timeout_s"]); return False
        fr = await _dock_frame("rear"); see = dock_see(fr.jpeg, "rear") if fr else None
        _dock["sense"] = see; _dock["cam"] = "rear"
        tag = see and see.get("tag"); sheet = see and see.get("sheet"); lane = see and see.get("lane_err")
        # steering reference, best first: tag pose (lateral offset + dock yaw) → tag bearing → sheet → rails
        if tag and tag.get("x_m") is not None and tag.get("z_m"):
            # error = where the dock axis crosses our path: lateral offset, plus the yaw we're
            # carrying - but only once the tag is big enough for the pose yaw to be stable
            # (flight log: at 45 px it flip-flopped +-30 deg every tick and made the rover fishtail)
            yaw_w = 0.8 if tag["side_px"] >= DOCK["yaw_min_px"] else 0.0
            x_err = max(-0.5, min(0.5, tag["x_m"] / max(0.3, tag["z_m"]) + math.radians(tag.get("yaw_deg", 0.0)) * yaw_w))
            ref = "tag %.2fm lat %+.2f yaw %+d°" % (tag["z_m"], tag["x_m"], tag.get("yaw_deg", 0))
            last_close = (tag["z_m"], tag["x_m"])
        elif tag:
            x_err = tag["x_err"]; ref = "tag %dpx" % tag["side_px"]
        elif sheet and sheet["ratio"] >= DOCK["min_ratio"]:
            x_err = sheet["x_err"]; ref = "sheet %d%%" % int(sheet["ratio"] * 100)
        elif lane is not None:
            x_err = lane; ref = "rails"
        else:
            x_err = None; ref = None
        if x_err is None and last_close and last_close[0] <= DOCK["final_z_m"] and abs(last_close[1]) <= 0.12:
            # the tag just slid out of view while we were close and aligned - that IS the last stretch
            return await _dock_final(last_close[0])
        if x_err is None:
            see = await _dock_settled_look("rear"); tag = see and see.get("tag"); sheet = see and see.get("sheet"); lane = see and see.get("lane_err")
            if tag and tag.get("x_m") is not None and tag.get("z_m"):
                x_err = max(-0.5, min(0.5, tag["x_m"] / max(0.3, tag["z_m"]) + math.radians(tag.get("yaw_deg", 0.0)) * 0.8)); ref = "tag %.2fm (settled)" % tag["z_m"]
            elif tag:
                x_err = tag["x_err"]; ref = "tag %dpx (settled)" % tag["side_px"]
            elif sheet and sheet["ratio"] >= DOCK["min_ratio"]:
                x_err = sheet["x_err"]; ref = "sheet (settled)"
            elif lane is not None:
                x_err = lane; ref = "rails (settled)"
        if x_err is None:
            _dock_set("lost", "dock not in view (rear) — sweeping")
            hit = await _dock_search("rear")
            if not hit:
                await _dock_send(0, 0); _dock["state"] = "failed"; _dock_set("no_target", "lost the dock while backing in and a full sweep found nothing"); return False
            cam, see = hit
            if cam == "front":
                _dock_set("turn_again", "dock is in front of us — turning around again"); return "turn"
            rev_since = None
            continue
        _dock["last_seen"] = now
        search_dir = 1.0 if x_err < 0 else -1.0
        near = bool(tag and (tag["side_px"] >= DOCK["tag_near_px"] or (tag.get("z_m") is not None and tag["z_m"] < 0.7))) or bool(sheet and sheet["ratio"] >= 0.45)
        angular = max(-DOCK["turn_max"], min(DOCK["turn_max"], learner.sign() * x_err * DOCK["turn_gain"]))
        if abs(x_err) > DOCK["turn_only"] and not near:
            linear = 0.0; _dock_set("align_rear", "lining up (rear) · " + ref); rev_since = None
        else:
            if abs(x_err) < DOCK["center_tol"]:
                angular = 0.0
            linear = -(DOCK["rev_near"] if near else DOCK["rev"])
            _dock_set("back", "backing onto the mat · " + ref)
            rev_since = rev_since or now
            if now - rev_since > DOCK["stall_s"] and _dock_rpms_zero():
                await _dock_send(0, 0); _dock["state"] = "docked"; _dock_set("docked", "wheels stalled against the stand"); return True
        learner.observe(x_err, angular != 0.0)
        await _dock_send(linear, angular)
        _dock_log_tick("back", ref)
        await asyncio.sleep(1.0 / DOCK["hz"])


def _dock_battery():
    try:
        b = (telemetry_hub.latest or {}).get("battery")
        return None if b is None else float(b)
    except Exception:
        return None


async def _dock_final(z_from: float):
    """Sep 5 (Cap: "rolling back very slowly until it receives charge - that's when it
    knows it's arrived"). The tag is gone from the rear frame because we are inside
    its field of view; roll straight back at DOCK_REV_FINAL until the wheels stall or
    the cap, then hold and watch the battery. Rising battery = docked and charging.
    No rise after DOCK_CHARGE_WAIT_S -> one more push, wait again, then report."""
    _dock_set("final", "tag out of view at %.2f m - rolling back slowly to the pads" % z_from)
    t0 = time.time(); stalled = False
    while time.time() - t0 < DOCK["final_max_s"]:
        await _dock_send(-DOCK["rev_final"], 0.0)
        _dock_log_tick("final", "blind roll")
        await asyncio.sleep(1.0 / DOCK["hz"])
        if time.time() - t0 > DOCK["stall_s"] and _dock_rpms_zero():
            stalled = True; break
    await _dock_send(0, 0)
    _dock_set("contact", "at the stand (%s) - watching for charge" % ("wheels stalled" if stalled else "rolled the full distance"))
    b0 = _dock_battery(); tw = time.time(); nudged = False
    while True:
        await asyncio.sleep(3.0)
        b = _dock_battery()
        _dock_log_tick("contact", "bat %s->%s" % (b0, b))
        if b0 is not None and b is not None and b >= b0 + 1:
            _dock["state"] = "docked"; _dock_set("docked", "charging - battery %d%% -> %d%%" % (b0, b)); _odo_set_home("self-dock charging"); return True
        if b0 is not None and b is not None and b0 >= 99:
            _dock["state"] = "docked"; _dock_set("docked", "on the pads at %d%% (battery full - can't see charge)" % b); return True
        if time.time() - tw > DOCK["charge_wait_s"]:
            if not nudged:
                nudged = True
                _dock_set("nudge", "no charge after %ds - one more push" % DOCK["charge_wait_s"])
                await _dock_send(-DOCK["rev_final"], 0.0); await asyncio.sleep(DOCK["nudge_s"]); await _dock_send(0, 0)
                tw = time.time(); b0 = _dock_battery(); continue
            _dock["state"] = "docked_nocharge"; _dock_set("no_charge", "on the stand but the battery is not rising - check the pads"); return False


# ★ OKCREAL (Sep 5, 2026 — Cap: "how we make the system automatically return to
#   this docking position… the rooms will change on every listing; the one thing
#   that stays the same is the dock — we power the rover on the first time already
#   on the dock"). RETURN TO DOCK = dead-reckoned breadcrumbs + the vision docker.
#   HOME is the dock: set when the rover is placed (SET HOME), when a self-dock
#   succeeds, or by Connect when it sees the battery charging. From then on every
#   drive command is integrated (commanded speed × time along the compass heading)
#   into a position relative to home, and a breadcrumb is dropped every
#   ODO_CRUMB_M along the way; loops in the trail are cut. RETURN walks the
#   crumbs back in reverse — the path it already drove, so doorways and furniture
#   are handled by the human who drove out — and when it is within
#   RETURN_HANDOFF_M of home, or either camera sees the dock, it hands off to the
#   self-docker. Dead reckoning drifts; the handoff distance covers that.
ODO = {
    "mps_per_unit": float(os.getenv("ODO_MPS_PER_UNIT", "1.1")),   # linear 1.0 ≈ 4 km/h (Frodobots spec)
    "crumb_m": float(os.getenv("ODO_CRUMB_M", "0.4")),
    "loop_cut_m": float(os.getenv("ODO_LOOP_CUT_M", "0.6")),
    "handoff_m": float(os.getenv("RETURN_HANDOFF_M", "2.5")),
    "reach_m": float(os.getenv("RETURN_REACH_M", "0.35")),
    "drive": float(os.getenv("RETURN_DRIVE", "0.18")),
    "turn": float(os.getenv("RETURN_TURN", "0.5")),
    "turn_only_deg": float(os.getenv("RETURN_TURN_ONLY_DEG", "28")),
    "crumb_timeout_s": float(os.getenv("RETURN_CRUMB_TIMEOUT_S", "25")),
    "timeout_s": float(os.getenv("RETURN_TIMEOUT_S", "480")),
    "look_every_s": float(os.getenv("RETURN_LOOK_EVERY_S", "4")),
}
_odo = {"home_set": False, "home_hdg": None, "x": 0.0, "y": 0.0, "crumbs": [], "last_cmd": (0.0, 0.0), "last_t": None, "dist_total": 0.0, "home_at": None}
_ret = {"task": None, "state": "idle", "phase": None, "reason": None, "since": None, "started_at": None, "target_i": None, "dist_home": None}


def _odo_pos():
    return {"x": round(_odo["x"], 2), "y": round(_odo["y"], 2), "dist_home": round(math.hypot(_odo["x"], _odo["y"]), 2), "crumbs": len(_odo["crumbs"]), "home_set": _odo["home_set"], "home_hdg": _odo["home_hdg"], "trail_m": round(_odo["dist_total"], 1)}


def _odo_integrate(now=None):
    """Advance position by the last command over the elapsed time (dead-man capped)."""
    now = now or time.time()
    lt = _odo["last_t"]
    if lt is None:
        _odo["last_t"] = now; return
    dt = min(max(0.0, now - lt), 1.0)
    _odo["last_t"] = now
    lin = _odo["last_cmd"][0]
    if not lin or not _odo["home_set"]:
        return
    h = _dock_heading()
    if h is None:
        return
    d = lin * ODO["mps_per_unit"] * dt
    _odo["x"] += d * math.sin(math.radians(h)); _odo["y"] += d * math.cos(math.radians(h))
    _odo["dist_total"] += abs(d)
    cr = _odo["crumbs"]
    if not cr or math.hypot(_odo["x"] - cr[-1][0], _odo["y"] - cr[-1][1]) >= ODO["crumb_m"]:
        # loop cut: if we are back near an earlier crumb, drop everything after it
        for i in range(len(cr) - 3, -1, -1):
            if math.hypot(_odo["x"] - cr[i][0], _odo["y"] - cr[i][1]) < ODO["loop_cut_m"]:
                del cr[i + 1:]; break
        cr.append((round(_odo["x"], 3), round(_odo["y"], 3), h))
        if len(cr) > 4000:
            del cr[0:len(cr) - 4000]


def _odo_note(cmd):
    try:
        _odo_integrate()
        _odo["last_cmd"] = (float(cmd.get("linear") or 0.0), float(cmd.get("angular") or 0.0))
    except Exception:
        pass


def _odo_set_home(reason: str):
    _odo.update({"home_set": True, "home_hdg": _dock_heading(), "x": 0.0, "y": 0.0, "crumbs": [], "last_cmd": (0.0, 0.0), "last_t": time.time(), "dist_total": 0.0, "home_at": time.time()})
    logger.info("odo: HOME set (%s) heading=%s", reason, _odo["home_hdg"])


async def _odo_ticker():
    while True:
        try:
            _odo_integrate()
        except Exception:
            pass
        await asyncio.sleep(0.5)


def _return_active():
    t = _ret["task"]; return bool(t and not t.done())


async def _return_cancel(reason):
    t = _ret["task"]
    if t and not t.done():
        _ret["reason"] = reason; t.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await t


def _ret_set(phase, reason=None):
    if _ret["phase"] != phase:
        _ret["phase"] = phase; _ret["since"] = time.time(); logger.info("return: %s%s", phase, (" — " + reason) if reason else "")
        _docklog_event("return", phase + ((" — " + reason) if reason else ""), {"pos": _odo_pos()})
    if reason:
        _ret["reason"] = reason


async def _return_dock_visible():
    """Settled look with both cameras: is the dock in view?"""
    for cam in ("front", "rear"):
        try:
            if cam == "rear" and not await browser_service.has_rear_camera():
                continue
            see = await _dock_settled_look(cam)
            tag = see and see.get("tag"); sheet = see and see.get("sheet")
            if tag or (sheet and sheet["ratio"] >= DOCK["sheet_confirm"]):
                return cam
        except Exception:
            pass
    return None


async def _return_loop():
    _ret.update({"state": "returning", "started_at": time.time(), "reason": None, "phase": None})
    _docklog_reset(); _docklog_event("start", "return to dock started", {"pos": _odo_pos()})
    try:
        if not _odo["home_set"]:
            _ret["state"] = "failed"; _ret_set("no_home", "HOME is not set — dock the rover once (or press SET HOME while docked)"); return
        crumbs = list(_odo["crumbs"]); crumbs.append((0.0, 0.0, _odo["home_hdg"]))   # …and finally home itself
        # walk the trail backwards, skipping crumbs we are already past
        targets = list(reversed(crumbs))
        spin_sign = _dock["sign"]["spin"]; last_look = 0.0; i = 0
        while i < len(targets):
            if time.time() - _ret["started_at"] > ODO["timeout_s"]:
                await _dock_send(0, 0); _ret["state"] = "failed"; _ret_set("timeout", "gave up after %ds" % ODO["timeout_s"]); return
            tx, ty, _ = targets[i]; _ret["target_i"] = i
            dist_home = math.hypot(_odo["x"], _odo["y"]); _ret["dist_home"] = round(dist_home, 2)
            # hand-off checks: close to home, or the dock is in view
            if dist_home <= ODO["handoff_m"]:
                await _dock_send(0, 0); _ret_set("handoff", "within %.1f m of home — vision docking takes over" % dist_home); break
            if time.time() - last_look > ODO["look_every_s"]:
                last_look = time.time()
                cam = await _return_dock_visible()
                if cam:
                    _ret_set("handoff", "the %s camera can see the dock — vision docking takes over" % cam); break
            # skip crumbs that are farther from home than we are (already passed) except the last few
            if i < len(targets) - 2 and math.hypot(tx, ty) > dist_home + ODO["crumb_m"]:
                i += 1; continue
            t_crumb = time.time()
            while True:
                dx = tx - _odo["x"]; dy = ty - _odo["y"]; d = math.hypot(dx, dy)
                if d <= ODO["reach_m"]:
                    break
                if time.time() - t_crumb > ODO["crumb_timeout_s"]:
                    _docklog_event("return", "crumb %d timed out at %.1f m — skipping" % (i, d)); break
                h = _dock_heading()
                if h is None:
                    await _dock_send(0, 0); _ret_set("wait", "no compass heading"); await asyncio.sleep(0.5); continue
                bearing = math.degrees(math.atan2(dx, dy)) % 360.0
                rel = ((bearing - h + 540.0) % 360.0) - 180.0
                if abs(rel) > ODO["turn_only_deg"]:
                    await _dock_send(0, ODO["turn"] * spin_sign * (-1 if rel < 0 else 1))
                    _ret_set("turn", "crumb %d/%d · %.1f m · turn %+d°" % (i, len(targets) - 1, d, rel))
                else:
                    ang = max(-0.35, min(0.35, spin_sign * rel / 40.0)) if abs(rel) > 6 else 0.0
                    await _dock_send(ODO["drive"], ang)
                    _ret_set("drive", "crumb %d/%d · %.1f m · home %.1f m" % (i, len(targets) - 1, d, math.hypot(_odo["x"], _odo["y"])))
                _dock_log_tick("return", "crumb %d d=%.2f rel=%+d" % (i, d, rel))
                await asyncio.sleep(0.25)
                # learn the spin sign from the first in-place turn: if |rel| grows, flip
                if abs(rel) > ODO["turn_only_deg"]:
                    await asyncio.sleep(0.6); h2 = _dock_heading()
                    if h2 is not None:
                        rel2 = ((bearing - h2 + 540.0) % 360.0) - 180.0
                        if abs(rel2) > abs(rel) + 3:
                            spin_sign *= -1; _dock["sign"]["spin"] = spin_sign; _docklog_event("sign_flip", "return spin sign → %+d" % int(spin_sign))
            i += 1
        await _dock_send(0, 0)
        _ret["state"] = "docking"; _ret_set("docking", "starting vision self-dock")
        await _dock_loop()
        _ret["state"] = "docked" if _dock["state"] == "docked" else ("failed" if _dock["state"] in ("failed", "docked_nocharge") else _dock["state"])
        _ret_set("done", "self-dock ended: " + str(_dock["state"]) + (" — " + str(_dock["reason"]) if _dock.get("reason") else ""))
        if _dock["state"] == "docked":
            _odo_set_home("docked after return")
    except asyncio.CancelledError:
        try:
            await _dock_send(0, 0)
        finally:
            _ret["state"] = "aborted"; _ret_set("aborted", _ret.get("reason") or "cancelled")
        raise
    except Exception as e:
        logger.error("return: loop error: %s", e)
        try:
            await _dock_send(0, 0)
        except Exception:
            pass
        _ret["state"] = "failed"; _ret_set("error", str(e)[:120])


def _return_status():
    return {"active": _return_active(), "state": _ret["state"], "phase": _ret["phase"], "reason": _ret["reason"], "dist_home": _ret["dist_home"],
            "elapsed_s": (round(time.time() - _ret["started_at"], 1) if _ret["started_at"] else None), "odo": _odo_pos()}


@app.get("/home")
async def home_get():
    return {"odo": _odo_pos(), "return": _return_status()}


@app.post("/home")
async def home_post(request: Request):
    body = await request.json()
    action = (body or {}).get("action", "set")
    if action == "set":
        _odo_set_home(str((body or {}).get("reason") or "manual"))
        return {"ok": True, "odo": _odo_pos()}
    if action == "clear":
        _odo.update({"home_set": False, "crumbs": [], "x": 0.0, "y": 0.0}); return {"ok": True}
    raise HTTPException(status_code=400, detail="action must be set|clear")


@app.post("/return")
async def return_post(request: Request):
    await need_start_mission()
    if not auth_response_data:
        await auth()
    body = await request.json()
    action = (body or {}).get("action", "start")
    if action == "stop":
        await _return_cancel("stopped by operator"); await _dock_cancel("stopped by operator")
        return _return_status()
    if _return_active() or _dock_active():
        return _return_status()
    _ret["task"] = asyncio.create_task(_return_loop())
    await asyncio.sleep(0.05)
    return _return_status()


async def _dock_loop():
    _dock.update({"state": "docking", "started_at": time.time(), "last_seen": None, "sense": None, "reason": None, "cmds": 0, "cam": None})
    _docklog_reset()
    _docklog_event("start", "self-dock started", {"mirror": dict(_dock["mirror"]), "sign": dict(_dock["sign"]), "heading": _dock_heading(), "battery": (telemetry_hub.latest or {}).get("battery")})
    _dock_set("acquire")
    try:
        if not await browser_service.has_rear_camera():
            _dock["state"] = "failed"; _dock_set("no_rear_cam", "this rover publishes no rear camera"); return
        ap = await _dock_stage_approach()
        if not ap:
            return
        _dock["last_seen"] = None
        for attempt in range(3):
            if ap != "back":
                if not await _dock_stage_turn():
                    return
                _dock["last_seen"] = None
            ap = True
            r = await _dock_stage_back()
            if r != "turn":
                return
        _dock["state"] = "failed"; _dock_set("turn_lost", "could not get the rear camera onto the dock after 3 turns")
    except asyncio.CancelledError:
        try:
            await _dock_send(0, 0)
        finally:
            if _dock["state"] == "docking":
                _dock["state"] = "aborted"; _dock_set("aborted", _dock.get("reason") or "cancelled")
        raise
    except Exception as e:
        logger.error("dock: loop error: %s", e)
        try:
            await _dock_send(0, 0)
        except Exception:
            pass
        _dock["state"] = "failed"; _dock_set("error", str(e)[:120])
    finally:
        _docklog_event("end", "run ended: " + str(_dock.get("state")) + (" — " + str(_dock.get("reason")) if _dock.get("reason") else ""),
                       {"mirror": dict(_dock["mirror"]), "sign": dict(_dock["sign"]), "cmds": _dock["cmds"]})


def _dock_active() -> bool:
    t = _dock["task"]
    return bool(t and not t.done())


async def _dock_cancel(reason: str):
    t = _dock["task"]
    if t and not t.done():
        _dock["reason"] = reason
        t.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await t


def _dock_progress():
    """0–100 for the console's progress bar, from phase + what the tag says."""
    ph = _dock["phase"] or ""; st = _dock["state"]; sn = _dock.get("sense") or {}; tag = sn.get("tag") or {}
    if st == "docked": return 100
    if ph in ("acquire",): return 3
    if ph in ("lost", "search"): return 8 if _dock["cam"] == "front" else 68
    if ph in ("stage", "approach", "align"):
        d = tag.get("stage_dist_m") if tag.get("stage_dist_m") is not None else (tag.get("z_m") if tag.get("z_m") is not None else None)
        if d is None: return 15
        return int(max(15, min(42, 42 - min(d, 4.0) / 4.0 * 27)))
    if ph in ("face", "square"): return 45
    if ph == "staged": return 50
    if ph == "turn":
        try:
            m = re.search(r"~(\d+)°", _dock.get("reason") or ""); t = float(m.group(1)) if m else 0.0
        except Exception:
            t = 0.0
        return int(50 + min(t, 180) / 180 * 15)
    if ph in ("turned", "turn_again"): return 65
    if ph in ("align_rear", "back"):
        z = tag.get("z_m")
        if z is None: return 70
        return int(max(70, min(90, 90 - min(max(z - 0.2, 0), 1.0) / 1.0 * 20)))
    if ph == "final": return 93
    if ph in ("contact", "nudge"): return 96
    return 0


def _dock_status():
    return {"return": _return_status(), "odo": _odo_pos(), "active": _dock_active(), "state": _dock["state"], "phase": _dock["phase"], "reason": _dock["reason"], "cam": _dock["cam"], "progress": _dock_progress(),
            "sense": _dock["sense"], "mirror": _dock["mirror"], "sign": _dock["sign"], "heading": _dock_heading(),
            "elapsed_s": (round(time.time() - _dock["started_at"], 1) if _dock["started_at"] else None),
            "phase_s": (round(time.time() - _dock["since"], 1) if _dock["since"] else None), "cmds": _dock["cmds"]}


@app.get("/dock")
async def dock_get():
    return _dock_status()


@app.get("/dock/log")
async def dock_log(snaps: int = 1):
    """Flight log of the last self-dock run (see _docklog)."""
    out = {"run_id": _docklog["run_id"], "generated_at": time.time(), "state": _dock_status(), "config": _docklog["config"],
           "events": _docklog["events"], "ticks": _docklog["ticks"], "snaps": _docklog["snaps"] if snaps else [],
           "bot": {"slug": os.getenv("BOT_SLUG"), "type": (auth_response_data or {}).get("BOT_TYPE")}}
    return JSONResponse(content=out)


@app.post("/dock")
async def dock_post(request: Request):
    await need_start_mission()
    if not auth_response_data:
        await auth()
    body = await request.json()
    action = (body or {}).get("action", "start")
    if action == "stop":
        await _dock_cancel("stopped by operator")
        return _dock_status()
    if action == "test":
        # one frame from each camera, no motion — what the docker sees right now
        out = {"front": None, "rear": None, "mirror": _dock["mirror"], "heading": _dock_heading()}
        for cam in ("front", "rear"):
            try:
                if cam == "rear" and not await browser_service.has_rear_camera():
                    out[cam] = {"error": "no rear camera"}; continue
                fr = await feed_broadcasters[cam].get_frame(max_age=0.5, timeout=3.0, fps=5)
                out[cam] = dock_see(fr.jpeg, cam) if fr else None
            except Exception as e:
                out[cam] = {"error": str(e)[:100]}
        return {"sense": out}
    if _dock_active():
        return _dock_status()
    _dock["task"] = asyncio.create_task(_dock_loop())
    await asyncio.sleep(0.05)
    return _dock_status()


@app.get("/rtc-config")
async def rtc_config():
    """★ OKCREAL (Sep 5, 2026 — Cap: "the Frodobots dashboard plays video awesome,
    can we not have that quality?"). The dashboard joins the Agora channel as a
    spectator and gets the rover's WebRTC stream directly — no screenshots,
    no JPEG hops. This hands a Connect console the same spectator credentials
    Frodobots issued to this SDK session so it can do exactly that."""
    await need_start_mission()
    if not auth_response_data:
        await auth()
    t = auth_response_data or {}
    uid = t.get("SPECTATOR_USERID"); tok = t.get("SPECTATOR_RTC_TOKEN")
    if not (t.get("APP_ID") and t.get("CHANNEL_NAME") and uid and tok):
        raise HTTPException(status_code=404, detail="no spectator token in this session")
    return {"app_id": t.get("APP_ID"), "channel": t.get("CHANNEL_NAME"), "uid": str(uid), "token": tok,
            "front_uid": 1000, "rear_uid": 1001, "bot_uid": t.get("BOT_UID")}


@app.post("/talk")
async def talk(request: Request):
    """Live voice from a console: {"pcm_b64": <s16le mono>, "rate": 16000} per chunk; {"stop": true} ends."""
    await need_start_mission()
    if not auth_response_data:
        await auth()
    body = await request.json()
    if body.get("stop"):
        return {"result": await browser_service.talk_stop()}
    b64 = body.get("pcm_b64")
    if not b64:
        raise HTTPException(status_code=400, detail="pcm_b64 required")
    try:
        chunks = await browser_service.talk_feed(b64, int(body.get("rate") or 16000))
        return {"ok": True, "chunks": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"talk failed: {str(e)[:120]}") from e


@app.get("/audio-status")
async def audio_status():
    try:
        page = await browser_service._run(lambda p: p.evaluate("() => window.audioTapStatus ? window.audioTapStatus() : null"), retry_on_disconnect=False)
    except Exception:
        page = None
    return {"listeners": audio_hub.listeners, "ingest": audio_hub.ingest_connected, "last_chunk_age_s": (round(time.time() - audio_hub.last_chunk_at, 1) if audio_hub.last_chunk_at else None), "page": page}


@app.get("/live-status")
async def live_status():
    try:
        return await browser_service.live_status() or {"playing": False}
    except Exception:
        return {"playing": False}


_speak_lock = asyncio.Lock()


@app.post("/speak")
async def speak(request: Request):
    await need_start_mission()
    if not auth_response_data:
        await auth()

    body = await request.json()
    text = body.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="Text not provided")

    # ★ OKCREAL (Sep 5, 2026 — Cap: "when i push more than one voice command the
    #   second one does not work"). Upstream wrote EVERY utterance to the same
    #   file (static/tts_output.mp3) and handed the browser the same URL each
    #   time, so a second command overwrote the file while the first was still
    #   being fetched/played and the page's fetch could serve the cached first
    #   clip. Fix: one unique file per utterance, commands queued one after
    #   another (never two audio tracks published at once), file removed after
    #   it has played.
    fname = f"tts_{int(time.time()*1000)}_{secrets.token_hex(3)}"
    audio_path = None
    try:
        audio_path = await generate_speech(text, f"static/{fname}")
        audio_filename = os.path.basename(audio_path)
        audio_url = f"http://127.0.0.1:8000/static/{audio_filename}?v={secrets.token_hex(4)}"
        async with _speak_lock:
            await browser_service.speak(audio_url)
        return {"message": "Speech sent to rover"}
    except Exception as e:
        logger.error("Error in /speak: %s", str(e))
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}") from e
    finally:
        if audio_path:
            try:
                os.remove(audio_path)
            except OSError:
                pass


@app.get("/screenshot")
async def get_screenshot(view_types: str = "rear,map,front"):
    await need_start_mission()
    if not auth_response_data:
        await auth()

    valid_views = {"rear", "map", "front"}
    views_list = view_types.split(",")

    for view in views_list:
        if view not in valid_views:
            raise HTTPException(status_code=400, detail=f"Invalid view type: {view}")

    screenshots = await browser_service.capture_screenshots(views_list)
    missing = [view for view in views_list if view not in screenshots]
    if missing:
        raise HTTPException(
            status_code=404, detail=f"Views not available: {', '.join(missing)}"
        )

    # Documented behavior since v3: the images are also saved to screenshots/.
    os.makedirs("screenshots", exist_ok=True)
    for view, image in screenshots.items():
        await asyncio.to_thread(
            browser_service._write_file,
            os.path.join("screenshots", f"{view}.png"),
            image,
        )

    response_content = {
        f"{view}_frame": base64.b64encode(image).decode("ascii")
        for view, image in screenshots.items()
    }

    response_content["timestamp"] = time.time()

    return JSONResponse(content=response_content)


@app.get("/data")
async def get_data():
    await need_start_mission()
    # Fast path: fresh telemetry pushed by the /sdk page, no page.evaluate.
    age = telemetry_hub.age_seconds
    if telemetry_hub.latest is not None and age is not None and age < 2:
        return JSONResponse(content=telemetry_hub.latest)
    data = await latest_rover_data()
    return JSONResponse(content=data)


def _pending_checkpoint_sequences() -> list[int]:
    sequences = sorted(
        cp.get("sequence")
        for cp in checkpoints_list_data.get("checkpoints_list", [])
        if isinstance(cp.get("sequence"), int)
    )
    try:
        latest = int(checkpoints_list_data.get("latest_scanned_checkpoint", 0))
    except (TypeError, ValueError):
        latest = 0
    return [sequence for sequence in sequences if sequence > latest]


@app.post("/checkpoint-reached")
async def checkpoint_reached(request: Request):
    global auth_response_data, checkpoints_list_data
    await need_start_mission()

    bot_slug = os.getenv("BOT_SLUG")
    mission_slug = os.getenv("MISSION_SLUG")
    auth_header = os.getenv("SDK_API_TOKEN")

    if not all([bot_slug, mission_slug, auth_header]):
        raise HTTPException(
            status_code=500, detail="Required environment variables not configured"
        )

    data = await latest_rover_data()
    latitude = data.get("latitude")
    longitude = data.get("longitude")

    if latitude is None or longitude is None:
        raise HTTPException(status_code=400, detail="Missing latitude or longitude")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_header}",
    }

    payload = {
        "bot_slug": bot_slug,
        "mission_slug": mission_slug,
        "latitude": latitude,
        "longitude": longitude,
    }

    # The backend ends the ride as part of accepting the final checkpoint.
    # Predict that transition from the cached mission progress and require a
    # confirmed zero while RTM is still available. With missing progress data,
    # be conservative because we cannot prove this is a non-final checkpoint.
    pending_sequences = _pending_checkpoint_sequences()
    checkpoint_may_complete = len(pending_sequences) <= 1
    if checkpoint_may_complete:
        await _require_confirmed_stop("complete the final checkpoint")

    status, response_data = await external_request(
        "POST",
        FRODOBOTS_API_URL + "/sdk/checkpoint_reached",
        headers=headers,
        json=payload,
    )
    if status != 200:
        raise HTTPException(
            status_code=status,
            detail={
                "error": response_data.get("error", "Failed to send checkpoint data"),
                "proximate_distance_to_checkpoint": response_data.get(
                    "distance_to_checkpoint", "Unknown"
                ),
            },
        )
    next_sequence = response_data.get("next_checkpoint_sequence", "")
    sequences = [
        cp.get("sequence")
        for cp in checkpoints_list_data.get("checkpoints_list", [])
        if isinstance(cp.get("sequence"), int)
    ]
    try:
        past_last = bool(sequences) and int(next_sequence) > max(sequences)
    except (TypeError, ValueError):
        past_last = False
    mission_completed = bool(sequences) and (not next_sequence or past_last)

    if pending_sequences:
        # Keep the cached progress current so the next request can identify the
        # final checkpoint before the backend tears down its RTM session.
        checkpoints_list_data["latest_scanned_checkpoint"] = pending_sequences[0]

    if mission_completed:
        # The backend ends the ride after the last checkpoint, which kills
        # the feed and makes the bot unreachable. Drop the local session so
        # /status reports it and /start-mission re-authenticates cleanly.
        # _require_confirmed_stop ran before the backend call for the final
        # checkpoint. It is now safe to tear down all local safety tasks.
        cancel_control_watchdog()
        auth_response_data = {}
        checkpoints_list_data = {}
        await asyncio.gather(
            *(broadcaster.close() for broadcaster in feed_broadcasters.values())
        )
        await browser_service.close()

    return JSONResponse(
        status_code=200,
        content={
            "message": "Checkpoint reached successfully",
            "next_checkpoint_sequence": next_sequence,
            "mission_completed": mission_completed,
        },
    )


@app.get("/missions-history")
async def missions_history():
    auth_header = os.getenv("SDK_API_TOKEN")
    bot_slug = os.getenv("BOT_SLUG")

    if not auth_header:
        raise HTTPException(status_code=500, detail="Authorization not configured")
    if not bot_slug:
        raise HTTPException(status_code=500, detail="Bot name not configured")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_header}",
    }

    data = {"bot_slug": bot_slug}

    status, response_data = await external_request(
        "POST", FRODOBOTS_API_URL + "/sdk/rides_history", headers=headers, json=data
    )
    if status != 200:
        raise HTTPException(
            status_code=status,
            detail="Failed to retrieve missions history",
        )
    return JSONResponse(content=response_data)


@app.get("/missions")
async def missions():
    auth_header = os.getenv("SDK_API_TOKEN")
    bot_slug = os.getenv("BOT_SLUG")

    if not auth_header:
        raise HTTPException(
            status_code=500, detail="Authorization header not configured"
        )
    if not bot_slug:
        raise HTTPException(status_code=500, detail="Bot name not configured")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_header}",
    }

    payload = {"bot_slug": bot_slug}

    status, response_data = await external_request(
        "GET", FRODOBOTS_API_URL + "/sdk/missions", headers=headers, params=payload
    )
    if status != 200:
        raise HTTPException(
            status_code=status,
            detail="Failed to retrieve missions",
        )
    missions_list = [
        {
            "slug": mission.get("slug"),
            "distance_in_m": mission.get("distance_in_m"),
            "checkpoints_count": mission.get("checkpoints_count"),
        }
        for mission in response_data.get("missions", [])
    ]
    return JSONResponse(content={"missions": missions_list})


@app.get("/v2/screenshot")
async def get_screenshot_v2():
    await need_start_mission()
    if not auth_response_data:
        await auth()

    async def get_frame(frame_type):
        frame, captured_at = await get_camera_frame(frame_type)
        if frame is None:
            return {}
        return {
            f"{frame_type}_frame": frame,
            f"{frame_type}_timestamp": captured_at,
        }

    front_task = asyncio.create_task(get_frame("front"))
    tasks = [front_task]

    if await browser_service.has_rear_camera():
        rear_task = asyncio.create_task(get_frame("rear"))
        tasks.append(rear_task)

    results = await asyncio.gather(*tasks)

    response_data = {}
    for result in results:
        response_data.update(result)

    if not response_data:
        raise HTTPException(status_code=404, detail="Frames not available")

    timestamps = [
        value for key, value in response_data.items() if key.endswith("_timestamp")
    ]
    response_data["timestamp"] = max(timestamps)

    return JSONResponse(content=response_data)


@app.get("/v2/front")
async def get_front_frame():
    await need_start_mission()
    if not auth_response_data:
        await auth()
    front_frame, captured_at = await get_camera_frame("front")
    response_data = {}
    if front_frame:
        response_data["front_frame"] = front_frame
        response_data["timestamp"] = captured_at
        return JSONResponse(content=response_data)
    else:
        raise HTTPException(status_code=404, detail="Front frame not available")


@app.get("/v2/rear")
async def get_rear_frame():
    await need_start_mission()
    if not auth_response_data:
        await auth()

    if not await browser_service.has_rear_camera():
        raise HTTPException(status_code=404, detail="Rear camera is not available")
    rear_frame, captured_at = await get_camera_frame("rear")
    response_data = {}
    if rear_frame:
        response_data["rear_frame"] = rear_frame
        response_data["timestamp"] = captured_at
        return JSONResponse(content=response_data)
    else:
        raise HTTPException(status_code=404, detail="Rear frame not available")


@app.post("/interventions/start")
async def start_intervention(request: Request):
    await need_start_mission()

    auth_header = os.getenv("SDK_API_TOKEN")
    bot_slug = os.getenv("BOT_SLUG")

    if not auth_header:
        raise HTTPException(
            status_code=500, detail="Authorization header not configured"
        )
    if not bot_slug:
        raise HTTPException(status_code=500, detail="Bot name not configured")

    data = await latest_rover_data()
    latitude = data.get("latitude")
    longitude = data.get("longitude")

    if latitude is None or longitude is None:
        raise HTTPException(status_code=400, detail="Missing latitude or longitude")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_header}",
    }

    payload = {
        "bot_slug": bot_slug,
        "latitude": latitude,
        "longitude": longitude,
    }

    status, response_data = await external_request(
        "POST",
        FRODOBOTS_API_URL + "/sdk/interventions/start",
        headers=headers,
        json=payload,
    )
    if status != 200:
        raise HTTPException(
            status_code=status,
            detail=response_data.get("error", "Failed to start intervention"),
        )
    return JSONResponse(
        status_code=200,
        content={
            "message": "Intervention started successfully",
            "intervention_id": response_data.get("intervention_id"),
        },
    )


@app.post("/interventions/end")
async def end_intervention(request: Request):
    await need_start_mission()

    auth_header = os.getenv("SDK_API_TOKEN")
    bot_slug = os.getenv("BOT_SLUG")

    if not auth_header:
        raise HTTPException(
            status_code=500, detail="Authorization header not configured"
        )
    if not bot_slug:
        raise HTTPException(status_code=500, detail="Bot name not configured")

    data = await latest_rover_data()
    latitude = data.get("latitude")
    longitude = data.get("longitude")

    if latitude is None or longitude is None:
        raise HTTPException(status_code=400, detail="Missing latitude or longitude")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_header}",
    }

    payload = {
        "bot_slug": bot_slug,
        "latitude": latitude,
        "longitude": longitude,
    }

    status, response_data = await external_request(
        "POST",
        FRODOBOTS_API_URL + "/sdk/interventions/end",
        headers=headers,
        json=payload,
    )
    if status != 200:
        raise HTTPException(
            status_code=status,
            detail=response_data.get("error", "Failed to end intervention"),
        )
    return JSONResponse(
        status_code=200,
        content={"message": "Intervention ended successfully"},
    )


@app.get("/interventions/history")
async def interventions_history():
    auth_header = os.getenv("SDK_API_TOKEN")
    bot_slug = os.getenv("BOT_SLUG")

    if not auth_header:
        raise HTTPException(
            status_code=500, detail="Authorization header not configured"
        )
    if not bot_slug:
        raise HTTPException(status_code=500, detail="Bot name not configured")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_header}",
    }

    payload = {"bot_slug": bot_slug}

    status, response_data = await external_request(
        "GET",
        FRODOBOTS_API_URL + "/sdk/interventions/history",
        headers=headers,
        params=payload,
    )
    if status != 200:
        raise HTTPException(
            status_code=status,
            detail="Failed to retrieve interventions history",
        )
    return JSONResponse(content=response_data)
