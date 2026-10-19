import json
import os
import unittest
from unittest.mock import patch

import main


class _FakeResponse:
    def __init__(self, status, body):
        self.status = status
        self._body = body

    async def json(self, content_type=None):
        return self._body

    async def text(self):
        return json.dumps(self._body)


class _FakeRequestContextManager:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    closed = False

    def __init__(self, response):
        self._response = response
        self.last_request = None

    def request(self, method, url, **kwargs):
        self.last_request = (method, url, kwargs)
        return _FakeRequestContextManager(self._response)


class ExternalRequestTest(unittest.IsolatedAsyncioTestCase):
    async def _call(self, status, body, debug):
        session = _FakeSession(_FakeResponse(status, body))
        env = {"DEBUG": "true"} if debug else {"DEBUG": "false"}
        with patch.dict(os.environ, env), \
                patch.object(main.app.state, "http_session", session, create=True), \
                patch.object(main, "logger") as mock_logger:
            result = await main.external_request("POST", "http://api/sdk/start_ride")
        return result, mock_logger

    async def test_returns_backend_status_and_body(self):
        error_body = {"error": "Bot is currently in use by another user"}
        (status, body), _ = await self._call(403, error_body, debug=False)

        self.assertEqual(status, 403)
        self.assertEqual(body, error_body)

    async def test_logs_real_backend_error_in_debug(self):
        error_body = {"error": "Bot is currently in use by another user"}
        _, mock_logger = await self._call(403, error_body, debug=True)

        mock_logger.error.assert_called_once()
        logged = str(mock_logger.error.call_args)
        self.assertIn("Bot is currently in use by another user", logged)
        self.assertIn("403", logged)

    async def test_does_not_log_error_without_debug(self):
        error_body = {"error": "Bot is currently in use by another user"}
        _, mock_logger = await self._call(403, error_body, debug=False)

        mock_logger.error.assert_not_called()

    async def test_does_not_log_error_on_success(self):
        _, mock_logger = await self._call(200, {"CHANNEL_NAME": "ride_1"}, debug=True)

        mock_logger.error.assert_not_called()


class SdkErrorDetailTest(unittest.TestCase):
    def test_maps_401_to_user_not_found(self):
        self.assertEqual(main._sdk_error_detail(401, "fallback"), "User not found")

    def test_maps_403_to_bot_unavailable(self):
        self.assertEqual(
            main._sdk_error_detail(403, "fallback"), "Bot unavailable for SDK"
        )

    def test_uses_fallback_for_other_statuses(self):
        self.assertEqual(main._sdk_error_detail(422, "fallback"), "fallback")
        self.assertEqual(main._sdk_error_detail(404, "fallback"), "fallback")
        self.assertEqual(main._sdk_error_detail(500, "fallback"), "fallback")

    def test_never_returns_raw_backend_text(self):
        # The mapping is keyed on status only, so backend internals can't leak.
        self.assertEqual(main._sdk_error_detail(422, "generic"), "generic")


if __name__ == "__main__":
    unittest.main()


class TokenErrorTest(unittest.IsolatedAsyncioTestCase):
    # ★ OKCREAL (Oct 19, 2026 — RV2b): the SDK remembers which FrodoBots call refused the bot (status only, never the body)
    #   so the rover browser's error and /status can say it; a later success clears it.

    async def _token(self, status, body):
        session = _FakeSession(_FakeResponse(status, body))
        with patch.object(main.app.state, "http_session", session, create=True):
            try:
                return await main.retrieve_tokens({"Authorization": "Bearer x"}, "volt-radon-lima")
            except main.HTTPException as exc:
                return exc

    async def test_refusal_is_remembered_without_the_body(self):
        exc = await self._token(404, {"error": "secret backend words"})
        self.assertEqual(exc.status_code, 404)
        self.assertEqual(exc.detail, "Bot unavailable for SDK")
        self.assertEqual(
            main.browser_service.token_error, "FrodoBots /sdk/token answered 404 for bot 'volt-radon-lima'"
        )
        self.assertNotIn("secret", main.browser_service.token_error)

    async def test_success_clears_it(self):
        await self._token(404, {"error": "x"})
        data = await self._token(200, {"CHANNEL_NAME": "c"})
        self.assertEqual(data, {"CHANNEL_NAME": "c"})
        self.assertIsNone(main.browser_service.token_error)

    async def test_no_answer_is_remembered(self):
        class _Down(_FakeSession):
            def request(self, method, url, **kwargs):
                raise main.aiohttp.ClientConnectionError("refused")

        with patch.object(main.app.state, "http_session", _Down(None), create=True):
            with self.assertRaises(main.HTTPException):
                await main.retrieve_tokens({}, "volt")
        self.assertEqual(
            main.browser_service.token_error,
            "FrodoBots /sdk/token did not answer (External API unavailable) for bot 'volt'",
        )
