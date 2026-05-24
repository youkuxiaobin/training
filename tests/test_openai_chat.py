import unittest

from openai_chat.client import OpenAIChatClient, OpenAIChatError, messages_to_prompt


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))


class FakeGenerator:
    tokenizer = FakeTokenizer()

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        use_cache: bool,
        include_prompt: bool,
    ) -> str:
        self.last_prompt = prompt
        self.last_max_new_tokens = max_new_tokens
        self.last_temperature = temperature
        self.last_top_k = top_k
        self.last_use_cache = use_cache
        self.last_include_prompt = include_prompt
        return "local reply"


class OpenAIChatClientTest(unittest.TestCase):
    def test_create_chat_completion_returns_openai_style_payload(self) -> None:
        generator = FakeGenerator()
        client = OpenAIChatClient(generator=generator, model_name="local-test")

        response = client.create_chat_completion(
            {
                "model": "local-test",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 8,
                "temperature": 0.1,
                "top_k": 5,
            }
        )

        self.assertEqual(response["object"], "chat.completion")
        self.assertEqual(response["model"], "local-test")
        self.assertEqual(response["choices"][0]["message"]["role"], "assistant")
        self.assertEqual(response["choices"][0]["message"]["content"], "local reply")
        self.assertEqual(generator.last_prompt, "User: hello\nAssistant:")
        self.assertEqual(generator.last_max_new_tokens, 8)
        self.assertEqual(generator.last_temperature, 0.1)
        self.assertEqual(generator.last_top_k, 5)
        self.assertFalse(generator.last_include_prompt)
        self.assertGreater(response["usage"]["total_tokens"], 0)

    def test_messages_to_prompt_preserves_roles(self) -> None:
        prompt = messages_to_prompt(
            [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "again"},
            ]
        )

        self.assertEqual(
            prompt,
            "System: be brief\nUser: hello\nAssistant: hi\nUser: again\nAssistant:",
        )

    def test_create_chat_completion_rejects_streaming(self) -> None:
        client = OpenAIChatClient(generator=FakeGenerator())

        with self.assertRaises(OpenAIChatError):
            client.create_chat_completion(
                {
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                }
            )


if __name__ == "__main__":
    unittest.main()
