#!/usr/bin/env python3
"""A scripted stand-in for the Gemini API, so a Sashiko review can run end to
end without a model or a key. It answers the path the Gemini provider posts
to, v1beta/models/<model>:generateContent, and the same for every request.

Every conversation that offers tools gets the same short script of tool
calls, chosen so both duplicate cases occur:

  turn 1: git_log(HEAD), git_show(HEAD)
  turn 2: git_log(HEAD)   repeat, not consecutive: reaches the ToolBox cache
  turn 3: git_log(HEAD)   repeat, consecutive: the guard refuses it (arm A)
  turn 4: final answer, a minimal instance of the requested JSON schema

Usage reports promptTokenCount ~ request_bytes/4, candidatesTokenCount 50,
thoughtsTokenCount 100 and a totalTokenCount that includes them, as Gemini
does, so the spend guard's use of total= can be checked.

  fake-gemini.py PORT [DELAY_SECONDS]

DELAY_SECONDS slows every answer, so a test can catch a run mid-flight.
"""
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SCRIPT = [
    [("git_log", {"range": "HEAD", "limit": 3}), ("git_show", {"object": "HEAD", "suppress_diff": True})],
    [("git_log", {"range": "HEAD", "limit": 3})],
    [("git_log", {"range": "HEAD", "limit": 3})],
]


def minimal(schema, defs):
    """The smallest value that satisfies a JSON schema's shape."""
    if not isinstance(schema, dict):
        return None
    if "$ref" in schema:
        return minimal(defs.get(schema["$ref"].split("/")[-1], {}), defs)
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema and schema[key]:
            return minimal(schema[key][0], defs)
    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]
    if "const" in schema:
        return schema["const"]
    t = schema.get("type")
    if isinstance(t, list):
        t = next((x for x in t if str(x).lower() != "null"), "null")
    t = str(t).lower() if t is not None else None
    if t == "object" or "properties" in schema:
        props = schema.get("properties", {})
        return {k: minimal(props.get(k, {}), defs) for k in schema.get("required", [])}
    if t == "array":
        n = schema.get("minItems", 0)
        return [minimal(schema.get("items", {}), defs) for _ in range(n)]
    if t == "string":
        return "x"
    if t in ("integer", "number"):
        return schema.get("minimum", 0)
    if t == "boolean":
        return False
    return None


def final_answer(body):
    cfg = body.get("generationConfig") or {}
    schema = cfg.get("responseSchema")
    if schema:
        defs = schema.get("$defs") or schema.get("definitions") or {}
        return json.dumps(minimal(schema, defs))
    if cfg.get("responseMimeType") == "text/plain":
        return "No issues found."
    return json.dumps({"concerns": [], "dismissed_concerns": []})


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        time.sleep(DELAY)
        body = json.loads(raw or b"{}")
        turns_with_calls = sum(
            1 for c in body.get("contents", [])
            if c.get("role") == "model" and any("functionCall" in p for p in c.get("parts", []))
        )
        if body.get("tools") and turns_with_calls < len(SCRIPT):
            parts = [{"functionCall": {"name": n, "args": a}} for n, a in SCRIPT[turns_with_calls]]
        else:
            parts = [{"text": final_answer(body)}]
        prompt = len(raw) // 4
        resp = {
            "candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}],
            "usageMetadata": {
                "promptTokenCount": prompt,
                "candidatesTokenCount": 50,
                "thoughtsTokenCount": 100,
                "totalTokenCount": prompt + 150,
            },
        }
        out = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


DELAY = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0

if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
