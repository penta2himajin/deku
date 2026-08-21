"""Red→green tests for the OpenAI-compatible completion client."""

from __future__ import annotations

import http.server
import json
import threading
import unittest

from deku import llm


class StubHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length)) if length else {}
        self.server.captured = {
            "path": self.path,
            "body": body,
            "authorization": self.headers.get("Authorization"),
        }
        payload = {
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "stub reply"},
                }
            ]
        }
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:  # quiet
        pass


class TestComplete(unittest.TestCase):
    def setUp(self) -> None:
        self.srv = http.server.HTTPServer(("127.0.0.1", 0), StubHandler)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.port = self.srv.server_port
        self._env = {
            "DEKU_URL": llm.BASE_URL,
            "DEKU_MODEL": llm.MODEL,
            "DEKU_API_KEY": llm.API_KEY,
        }
        llm.BASE_URL = f"http://127.0.0.1:{self.port}"
        llm.MODEL = "MiniCPM5-1B"
        llm.API_KEY = ""

    def tearDown(self) -> None:
        llm.BASE_URL = self._env["DEKU_URL"]
        llm.MODEL = self._env["DEKU_MODEL"]
        llm.API_KEY = self._env["DEKU_API_KEY"]
        self.srv.shutdown()
        self.srv.server_close()

    def test_chat_completions_path_and_messages(self) -> None:
        out = llm.complete("hello", system="be brief", max_tokens=32, temperature=0.0)
        self.assertEqual(out, "stub reply")
        self.assertEqual(self.srv.captured["path"], "/v1/chat/completions")
        body = self.srv.captured["body"]
        self.assertEqual(body["model"], "MiniCPM5-1B")
        self.assertIs(body["stream"], False)
        self.assertEqual(body["max_tokens"], 32)
        self.assertEqual(body["temperature"], 0.0)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(
            body["messages"],
            [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hello"},
            ],
        )

    def test_think_true_sets_enable_thinking(self) -> None:
        llm.complete("q", think=True)
        self.assertEqual(
            self.srv.captured["body"]["chat_template_kwargs"],
            {"enable_thinking": True},
        )

    def test_prompt_only_omits_system(self) -> None:
        llm.complete("just user")
        roles = [m["role"] for m in self.srv.captured["body"]["messages"]]
        self.assertEqual(roles, ["user"])

    def test_api_key_header(self) -> None:
        llm.API_KEY = "secret-token"
        llm.complete("q")
        self.assertEqual(
            self.srv.captured["authorization"], "Bearer secret-token"
        )

    def test_unreachable_raises(self) -> None:
        llm.BASE_URL = "http://127.0.0.1:1"
        with self.assertRaises(llm.LLMError):
            llm.complete("q", timeout=0.5)


    def test_prefill_uses_completions(self) -> None:
        class PrefillHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length)) if length else {}
                self.server.captured = {"path": self.path, "body": body}
                payload = {"choices": [{"index": 0, "text": "Paris", "finish_reason": "stop"}]}
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            def log_message(self, *args):
                pass

        self.srv.shutdown()
        self.srv.server_close()
        self.srv = http.server.HTTPServer(("127.0.0.1", 0), PrefillHandler)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        llm.BASE_URL = f"http://127.0.0.1:{self.srv.server_port}"
        out = llm.complete("q", prefill="ANSWER: ", max_tokens=8, temp=0.2, seed=1)
        self.assertEqual(out, "ANSWER: Paris")
        self.assertEqual(self.srv.captured["path"], "/v1/completions")
        self.assertTrue(self.srv.captured["body"]["prompt"].endswith("ANSWER: "))


if __name__ == "__main__":
    unittest.main()
