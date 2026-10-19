import unittest

from playwright.async_api import Error as PlaywrightError

from browser_service import BrowserService


class FakePage:
    def __init__(self, closed=False):
        self.closed = closed
        self.close_calls = 0

    def is_closed(self):
        return self.closed

    async def close(self):
        self.close_calls += 1
        self.closed = True


class FakeBrowser:
    def __init__(self, connected=True):
        self.connected = connected

    def is_connected(self):
        return self.connected

    async def close(self):
        self.connected = False


class BrowserRecoveryTest(unittest.IsolatedAsyncioTestCase):
    async def test_stale_failure_cannot_destroy_new_generation(self):
        service = BrowserService()
        stale = FakePage(closed=True)
        current = FakePage()
        service._page = current
        service._browser = FakeBrowser()
        service._ready = True

        await service._invalidate(stale)

        self.assertIs(service._page, current)
        self.assertEqual(current.close_calls, 0)

    async def test_application_error_does_not_destroy_healthy_browser(self):
        service = BrowserService()
        page = FakePage()
        service._page = page
        service._browser = FakeBrowser()
        service._ready = True

        async def fail(_page):
            raise PlaywrightError("JavaScript application error")

        with self.assertRaises(PlaywrightError):
            await service._run(fail)

        self.assertIs(service._page, page)
        self.assertEqual(page.close_calls, 0)

    async def test_page_is_not_ready_until_launch_finishes(self):
        service = BrowserService()
        service._page = FakePage()
        service._browser = FakeBrowser()

        self.assertFalse(service.is_ready)

        service._ready = True
        self.assertTrue(service.is_ready)


class RearCameraCacheTest(unittest.IsolatedAsyncioTestCase):
    # /v2/screenshot consults has_rear_camera() on every poll; the answer
    # must come from cache, not a page.evaluate round trip per request.

    def _service(self, result):
        service = BrowserService()
        service._page = FakePage()
        service._browser = FakeBrowser()
        service._ready = True
        service.run_calls = 0

        async def run(fn, **kwargs):
            service.run_calls += 1
            return result

        service._run = run
        return service

    async def test_positive_result_is_cached_until_teardown(self):
        service = self._service(True)
        self.assertTrue(await service.has_rear_camera())
        self.assertTrue(await service.has_rear_camera())
        self.assertEqual(service.run_calls, 1)

        await service._teardown()  # relaunch must re-detect capability
        self.assertTrue(await service.has_rear_camera())
        self.assertEqual(service.run_calls, 2)

    async def test_negative_result_expires_quickly(self):
        import time

        service = self._service(False)
        self.assertFalse(await service.has_rear_camera())
        self.assertFalse(await service.has_rear_camera())
        self.assertEqual(service.run_calls, 1)

        # The rear track can subscribe late after the Agora join, so a
        # negative answer only holds for a short TTL.
        service._rear_camera = (False, time.monotonic() - 1)
        self.assertFalse(await service.has_rear_camera())
        self.assertEqual(service.run_calls, 2)


class LockLoopBindingTest(unittest.IsolatedAsyncioTestCase):
    async def test_lock_is_rebound_when_the_event_loop_changes(self):
        # The service is constructed at import time, before hypercorn's loop
        # exists. On Python 3.9 a Lock made then is bound to the wrong loop
        # and raises "attached to a different loop" under contention.
        import asyncio

        from browser_service import BrowserService

        service = BrowserService()
        service._lock = asyncio.Lock()
        service._lock_loop = object()  # simulates a dead import-time loop
        stale = service._lock

        lock = service._get_lock()
        self.assertIsNot(lock, stale)
        # Stable within the same running loop
        self.assertIs(service._get_lock(), lock)
        async with service._get_lock():
            pass  # must be acquirable on this loop


if __name__ == "__main__":
    unittest.main()


class LaunchDiagnosisTest(unittest.TestCase):
    # ★ OKCREAL (Oct 19, 2026 — RV2): Connect showed only the first line of Playwright's error. The diagnosis keeps the lines
    #   that explain a failed join and, when Chrome itself died, the container's memory and /dev/shm.

    def test_chrome_death_reports_host_limits_and_chrome_lines(self):
        from browser_service import launch_diagnosis

        err = (
            "BrowserContext.new_page: Target page, context or browser has been closed\n"
            "Browser logs:\n\n<launching> /opt/google/chrome/chrome --headless\n"
            "[pid=41][err] [41:41:FATAL:shared_memory_switch.cc(237)] Creating shared memory in /dev/shm failed: No space left on device\n"
            "[pid=41] <process did exit: exitCode=null, signal=SIGTRAP>"
        )
        d = launch_diagnosis(err, [])
        self.assertTrue(d.startswith("BrowserContext.new_page: Target page, context or browser has been closed"))
        self.assertIn("chrome: ", d)
        self.assertIn("Creating shared memory in /dev/shm failed", d)
        self.assertIn("host: ", d)
        self.assertLessEqual(len(d), 600)

    def test_failed_join_keeps_what_the_page_said(self):
        from browser_service import launch_diagnosis

        console = [
            "log: on connection state changed to CONNECTING reason: LOGIN",
            "log: AgoraRTM client login failure RtmUnauthorizedError: token expired",
            "log: on connection state changed to ABORTED reason: LOGIN_FAILURE",
            "log: harmless line",
        ]
        d = launch_diagnosis("Page.wait_for_function: Timeout 15000ms exceeded.\nCall log:\n  - waiting", console)
        self.assertTrue(d.startswith("Page.wait_for_function: Timeout 15000ms exceeded."))
        self.assertIn("page said: ", d)
        self.assertIn("RtmUnauthorizedError: token expired", d)
        self.assertNotIn("harmless line", d)
        self.assertNotIn("host: ", d)   # Chrome did not die; its limits are not the story

    def test_host_facts_text_reads_the_container(self):
        from browser_service import host_facts, host_facts_text

        f = host_facts()
        self.assertIn("cpus", f)
        self.assertTrue(host_facts_text(f))


class LaunchingFlagTest(unittest.IsolatedAsyncioTestCase):
    async def test_failed_launch_clears_launching_and_records_diagnosis(self):
        import browser_service as bs

        class Boom:
            async def start(self):
                raise PlaywrightError("BrowserType.launch: Target page, context or browser has been closed")

        service = BrowserService()
        seen = []
        orig = bs.async_playwright
        bs.async_playwright = lambda: Boom()
        try:
            async def watch():
                seen.append(service.launching)
            with self.assertRaises(PlaywrightError):
                task = service._launch()
                await watch()
                await task
        finally:
            bs.async_playwright = orig
        self.assertFalse(service.launching)
        self.assertIsNotNone(service.last_error_at)
        self.assertIn("host: ", service.last_error)


class ConsoleLineTest(unittest.TestCase):
    def test_one_line_without_styling(self):
        from browser_service import _console_line

        self.assertEqual(
            _console_line('log: AgoraRTM client login failure RtmTimeoutError:  Error Code 9 - timed out after 12s.\n    at t.<anonymous> (http://x/static/a.js)'),
            "log: AgoraRTM client login failure RtmTimeoutError:  Error Code 9 - timed out after 12s.",
        )
        self.assertEqual(
            _console_line("warning: 07:22:36:335 %cAgora-SDK [WARNING]: color: #FB8C00; font-weight: bold; [client-33994]"),
            "warning: 07:22:36:335 Agora-SDK [WARNING]: [client-33994]",
        )
