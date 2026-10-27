import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from evalgen.providers import CachedProvider, OllamaProvider, OpenAICompatibleProvider, ProviderError, ScriptedProvider


def fake_response(status=200, payload=None, text=""):
    r = mock.Mock()
    r.status_code = status
    r.json.return_value = payload or {}
    r.text = text
    return r


class TestOllama(unittest.TestCase):
    def test_payload_pins_decoding(self):
        p = OllamaProvider(base_url="http://h:1/", temperature=0.0, seed=7)
        with mock.patch("evalgen.providers.requests.post") as post:
            post.return_value = fake_response(payload={"message": {"content": "{}"}, "eval_count": 3})
            res = p.chat("m", [{"role": "user", "content": "hi"}], json_schema={"type": "object"})
        url = post.call_args.args[0]
        body = post.call_args.kwargs["json"]
        self.assertEqual(url, "http://h:1/api/chat")
        self.assertFalse(body["stream"])
        self.assertEqual(body["options"], {"temperature": 0.0, "seed": 7})
        self.assertEqual(body["format"], {"type": "object"})
        self.assertEqual(res.text, "{}")
        self.assertEqual(res.completion_tokens, 3)

    def test_http_error(self):
        with mock.patch("evalgen.providers.requests.post", return_value=fake_response(500, text="boom")):
            with self.assertRaises(ProviderError):
                OllamaProvider().chat("m", [])

    def test_release_unloads(self):
        with mock.patch("evalgen.providers.requests.post") as post:
            OllamaProvider(base_url="http://h:1").release("m")
        self.assertEqual(post.call_args.kwargs["json"], {"model": "m", "keep_alive": 0})


class TestOpenAICompatible(unittest.TestCase):
    def test_env_config_and_payload(self):
        env = {"EVALGEN_OPENAI_BASE_URL": "https://example.invalid/v1", "EVALGEN_OPENAI_API_KEY": "k"}
        with mock.patch.dict(os.environ, env), mock.patch("evalgen.providers.requests.post") as post:
            post.return_value = fake_response(payload={"choices": [{"message": {"content": "ok"}}],
                                                       "usage": {"prompt_tokens": 1, "completion_tokens": 2}})
            res = OpenAICompatibleProvider(seed=3).chat("gpt-x", [{"role": "user", "content": "q"}], {"type": "object"})
        self.assertEqual(post.call_args.args[0], "https://example.invalid/v1/chat/completions")
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer k")
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["seed"], 3)
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertEqual(res.text, "ok")

    def test_missing_env_is_an_error(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ProviderError):
                OpenAICompatibleProvider()


class TestCache(unittest.TestCase):
    def test_hits_and_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.sqlite"
            inner = ScriptedProvider(lambda m, msgs, s: f"reply to {msgs[-1]['content']}")
            c = CachedProvider(inner, path)
            a = c.chat("m", [{"role": "user", "content": "x"}])
            b = c.chat("m", [{"role": "user", "content": "x"}])
            d = c.chat("m", [{"role": "user", "content": "y"}])
            self.assertEqual(len(inner.calls), 2)
            self.assertFalse(a.cached)
            self.assertTrue(b.cached)
            self.assertEqual(b.text, a.text)
            self.assertEqual(d.text, "reply to y")
            c.close()
            c2 = CachedProvider(inner, path)
            self.assertTrue(c2.chat("m", [{"role": "user", "content": "x"}]).cached)
            self.assertEqual(len(inner.calls), 2)
            c2.close()


if __name__ == "__main__":
    unittest.main()
