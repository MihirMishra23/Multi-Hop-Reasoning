#!/usr/bin/env python3
"""Tiny official-schema IRCoT endpoint for end-to-end smoke tests only.

This is not BM25 and must never be used for a reported evaluation.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


DOCUMENTS = [
    {
        "id": "smoke-france",
        "title": "France",
        "paragraph_text": "France is a country in Western Europe. Its capital is Paris.",
        "paragraph_index": 0,
        "is_abstract": True,
        "url": "",
    },
    {
        "id": "smoke-paris",
        "title": "Paris",
        "paragraph_text": "Paris is the capital and most populous city of France.",
        "paragraph_index": 0,
        "is_abstract": True,
        "url": "",
    },
]


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length))
        corpus_name = request["corpus_name"]
        top_k = int(request["max_hits_count"])
        retrieval = [{**document, "corpus_name": corpus_name} for document in DOCUMENTS[:top_k]]
        content = json.dumps({"retrieval": retrieval, "time_in_seconds": 0.0}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    args = parser.parse_args()
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
