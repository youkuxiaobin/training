import json
import unittest
from unittest.mock import patch

from openai_chat.client import OpenAIChatClient, OpenAIChatError


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class OpenAIChatClientTest(unittest.TestCase):
    def test_create_reply_posts_responses_payload(self) -> None:
        captured = {}

        def fake_transport(request, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            captured["headers"] = dict(request.header_items())
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeHTTPResponse({"output_text": "hello back"})

        client = OpenAIChatClient(
            api_key="test-key",
            model="gpt-test",
            timeout=12,
            transport=fake_transport,
        )

        reply = client.create_reply(
            [{"role": "user", "content": "hello"}],
            instructions="Be brief.",
            max_output_tokens=32,
        )

        self.assertEqual(reply, "hello back")
        self.assertEqual(captured["url"], "https://api.openai.com/v1/responses")
        self.assertEqual(captured["timeout"], 12)
        self.assertEqual(captured["payload"]["model"], "gpt-test")
        self.assertEqual(captured["payload"]["instructions"], "Be brief.")
        self.assertEqual(captured["payload"]["max_output_tokens"], 32)
        self.assertEqual(captured["payload"]["input"], [{"role": "user", "content": "hello"}])
        self.assertEqual(captured["headers"]["Authorization"], "Bearer test-key")

    def test_create_reply_requires_api_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            client = OpenAIChatClient(api_key="", transport=lambda request, timeout: None)

            with self.assertRaises(OpenAIChatError):
                client.create_reply([{"role": "user", "content": "hello"}])


if __name__ == "__main__":
    unittest.main()
