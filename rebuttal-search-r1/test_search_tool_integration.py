"""Integration test: does verl's SearchTool actually hit the retrieval server?

This is the "given a valid tool call, does search fire?" check the training run
keeps failing — independent of whether the model generates the right format.

It bypasses the LLM entirely: we hand-construct a valid tool call (the exact
arguments dict verl would parse out of `<tool_call>{"name":"search",
"arguments":{"query_list":[...]}}</tool_call>`) and assert that:

  1. The retrieval server at :8000 is reachable and speaks the /retrieve schema.
  2. verl's low-level helper `perform_single_search_batch` parses the response
     into a Doc-formatted string.
  3. verl's `SearchTool.execute` (Ray actor + async wrapper) returns a
     ToolResponse whose text contains the retrieved documents, with
     metrics.status == "success" and total_results > 0.

Run (retrieval server must be up first):
    bash rebuttal-search-r1/launch_retrieval_verl.sh &      # in another shell
    python rebuttal-search-r1/test_search_tool_integration.py

Env overrides:
    SEARCH_URL=http://127.0.0.1:8000/retrieve
    TOOL_CONFIG=/path/to/search_tool_config.yaml
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import requests


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TOOL_CONFIG = SCRIPT_DIR / "verl_config" / "tool_config" / "search_tool_config.yaml"
DEFAULT_SEARCH_URL = "http://127.0.0.1:8000/retrieve"

# HotpotQA-style multi-hop queries — chosen so that the wiki-18 corpus should
# return non-empty top-k results.
SAMPLE_QUERIES = [
    "Akira Kurosawa film director birth year",
    "Stanley Kubrick film director birth year",
]


def _hdr(title):
    bar = "=" * 78
    print(f"\n{bar}\n{title}\n{bar}", flush=True)


def _ok(msg):  print(f"  [ OK ] {msg}", flush=True)
def _fail(msg):print(f"  [FAIL] {msg}", flush=True)
def _info(msg):print(f"  [info] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — server is up and speaks the /retrieve protocol verl expects.
# ─────────────────────────────────────────────────────────────────────────────
def test_server_reachable(search_url: str) -> bool:
    _hdr("Test 1: retrieval server health + raw /retrieve protocol")

    base = search_url.rsplit("/", 1)[0]  # strip /retrieve
    health_url = f"{base}/health"

    try:
        r = requests.get(health_url, timeout=5)
        r.raise_for_status()
        _ok(f"GET {health_url} -> {r.status_code} {r.text.strip()}")
    except Exception as e:
        _fail(f"GET {health_url} failed: {e!r}")
        _info("Start the server first: bash rebuttal-search-r1/launch_retrieval_verl.sh")
        return False

    payload = {"queries": SAMPLE_QUERIES, "topk": 3, "return_scores": True}
    try:
        t0 = time.time()
        r = requests.post(search_url, json=payload, timeout=60)
        dt = time.time() - t0
        r.raise_for_status()
        data = r.json()
        _ok(f"POST {search_url} -> {r.status_code} ({dt:.2f}s)")
    except Exception as e:
        _fail(f"POST {search_url} failed: {e!r}")
        return False

    # Schema check — verl expects {"result": [ [ {"document": {...}, "score": float}, ... ], ... ]}
    if "result" not in data or not isinstance(data["result"], list):
        _fail(f"response missing top-level 'result' list: keys={list(data)}")
        return False
    if len(data["result"]) != len(SAMPLE_QUERIES):
        _fail(f"got {len(data['result'])} result groups, expected {len(SAMPLE_QUERIES)}")
        return False
    for i, group in enumerate(data["result"]):
        if not isinstance(group, list) or len(group) == 0:
            _fail(f"query[{i}] returned empty / non-list result group")
            return False
        first = group[0]
        if not isinstance(first, dict) or "document" not in first:
            _fail(f"query[{i}] doc missing 'document' field: {first!r}")
            return False
        doc = first["document"]
        if "contents" not in doc:
            _fail(f"query[{i}] document missing 'contents' field: keys={list(doc)}")
            return False
        _ok(f"query[{i}] returned {len(group)} docs; top contents[:80]={doc['contents'][:80]!r}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — verl's perform_single_search_batch (no Ray, just HTTP + formatting).
# ─────────────────────────────────────────────────────────────────────────────
def test_perform_single_search_batch(search_url: str) -> bool:
    _hdr("Test 2: verl.tools.utils.search_r1_like_utils.perform_single_search_batch")
    try:
        from verl.tools.utils.search_r1_like_utils import perform_single_search_batch
    except ImportError as e:
        _fail(f"cannot import perform_single_search_batch: {e!r}")
        return False

    result_text, metadata = perform_single_search_batch(
        retrieval_service_url=search_url,
        query_list=SAMPLE_QUERIES,
        topk=3,
        timeout=30,
    )

    _info(f"metadata.status        = {metadata.get('status')}")
    _info(f"metadata.query_count   = {metadata.get('query_count')}")
    _info(f"metadata.total_results = {metadata.get('total_results')}")
    _info(f"metadata.api_request_error = {metadata.get('api_request_error')}")

    if metadata.get("status") != "success":
        _fail(f"expected status=success, got {metadata.get('status')!r}")
        return False
    if metadata.get("total_results", 0) <= 0:
        _fail(f"expected total_results > 0, got {metadata.get('total_results')}")
        return False

    try:
        parsed = json.loads(result_text)
    except json.JSONDecodeError as e:
        _fail(f"result_text is not valid JSON: {e!r}")
        return False
    if "result" not in parsed:
        _fail(f"parsed result_text has no 'result' key: keys={list(parsed)}")
        return False
    formatted = parsed["result"]
    if "Doc 1 (Title:" not in formatted:
        _fail(f"formatted result missing 'Doc 1 (Title:' marker:\n{formatted[:400]}")
        return False
    _ok(f"got formatted Doc-string ({len(formatted)} chars), preview:")
    print("    " + formatted[:400].replace("\n", "\n    "))
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — full SearchTool path: load from yaml, create instance, execute().
# This is the exact code path the rollout uses when a model emits <tool_call>.
# ─────────────────────────────────────────────────────────────────────────────
def test_search_tool_execute(tool_config_path: Path, search_url: str) -> bool:
    _hdr("Test 3: verl.tools.search_tool.SearchTool.execute (full Ray path)")
    if not tool_config_path.exists():
        _fail(f"tool config not found: {tool_config_path}")
        return False

    try:
        import ray
        from verl.tools.utils.tool_registry import initialize_tools_from_config
        from verl.tools.schemas import ToolResponse
    except ImportError as e:
        _fail(f"cannot import verl tool modules: {e!r}")
        return False

    if not ray.is_initialized():
        # local_mode=True keeps everything in the driver process — fast & easy
        # to debug. Set RAY_LOCAL_MODE=0 to force a real cluster if needed.
        local_mode = os.environ.get("RAY_LOCAL_MODE", "1") == "1"
        _info(f"ray.init(local_mode={local_mode})")
        ray.init(local_mode=local_mode, ignore_reinit_error=True, log_to_driver=False)

    tools = initialize_tools_from_config(str(tool_config_path))
    if not tools:
        _fail("initialize_tools_from_config returned no tools")
        return False
    tool = tools[0]
    _ok(f"loaded tool: {tool.__class__.__name__} (url={getattr(tool, 'retrieval_service_url', '?')})")

    # The yaml might point at a different URL than the test arg — surface that.
    if getattr(tool, "retrieval_service_url", None) != search_url:
        _info(f"NOTE: tool URL from yaml = {tool.retrieval_service_url!r}, "
              f"--search-url = {search_url!r}")

    schema = tool.get_openai_tool_schema()
    fn_name = schema.function.name
    if fn_name != "search":
        _fail(f"expected tool name 'search', got {fn_name!r}")
        return False
    _ok(f"tool schema function.name == {fn_name!r}")

    async def _run():
        instance_id, create_resp = await tool.create()
        _ok(f"create() -> instance_id={instance_id[:8]}…")

        # Simulate the exact dict verl would pass after parsing a valid
        # <tool_call>{"name":"search","arguments":{"query_list":[...]}}</tool_call>
        parameters = {"query_list": SAMPLE_QUERIES}
        response, reward, metrics = await tool.execute(instance_id, parameters)
        return instance_id, response, reward, metrics

    try:
        instance_id, response, reward, metrics = asyncio.run(_run())
    except Exception as e:
        import traceback
        _fail(f"tool.execute() raised: {e!r}")
        traceback.print_exc()
        return False

    _info(f"reward          = {reward}")
    _info(f"metrics         = {metrics}")
    _info(f"response.text[:200] = {(response.text or '')[:200]!r}")

    if not isinstance(response, ToolResponse):
        _fail(f"expected ToolResponse, got {type(response).__name__}")
        return False
    if not response.text:
        _fail("ToolResponse.text is empty")
        return False
    if metrics.get("status") != "success":
        _fail(f"metrics.status != success ({metrics.get('status')!r}); "
              f"api_request_error={metrics.get('api_request_error')}")
        return False
    if metrics.get("total_results", 0) <= 0:
        _fail(f"metrics.total_results <= 0 ({metrics.get('total_results')})")
        return False

    try:
        body = json.loads(response.text)
    except json.JSONDecodeError as e:
        _fail(f"ToolResponse.text not valid JSON: {e!r}")
        return False
    formatted = body.get("result", "")
    if "Doc 1 (Title:" not in formatted:
        _fail(f"formatted result missing 'Doc 1 (Title:' marker:\n{formatted[:400]}")
        return False
    _ok("ToolResponse contains formatted docs:")
    print("    " + formatted[:400].replace("\n", "\n    "))
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — malformed tool call -> error, no crash (defensive sanity).
# ─────────────────────────────────────────────────────────────────────────────
def test_search_tool_invalid_args(tool_config_path: Path) -> bool:
    _hdr("Test 4: SearchTool.execute with invalid args (must NOT raise)")
    try:
        import ray
        from verl.tools.utils.tool_registry import initialize_tools_from_config
    except ImportError as e:
        _fail(f"cannot import verl tool modules: {e!r}")
        return False

    if not ray.is_initialized():
        ray.init(local_mode=True, ignore_reinit_error=True, log_to_driver=False)
    tool = initialize_tools_from_config(str(tool_config_path))[0]

    async def _run(params):
        instance_id, _ = await tool.create()
        return await tool.execute(instance_id, params)

    cases = [
        ("missing query_list", {}),
        ("query_list is not a list", {"query_list": "single string"}),
        ("query_list empty", {"query_list": []}),
    ]
    all_ok = True
    for label, params in cases:
        try:
            response, reward, metrics = asyncio.run(_run(params))
            text = response.text or ""
            if "Error" in text or "missing" in text.lower():
                _ok(f"{label!r:>30} -> graceful error: {text[:120]}")
            else:
                _fail(f"{label!r:>30} -> unexpected success: {text[:120]}")
                all_ok = False
        except Exception as e:
            _fail(f"{label!r:>30} -> raised {e!r}")
            all_ok = False
    return all_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-url", default=os.environ.get("SEARCH_URL", DEFAULT_SEARCH_URL),
                    help="POST endpoint that speaks verl's /retrieve protocol")
    ap.add_argument("--tool-config", default=os.environ.get("TOOL_CONFIG", str(DEFAULT_TOOL_CONFIG)),
                    help="Path to verl tool_config yaml")
    ap.add_argument("--skip-ray", action="store_true",
                    help="Skip Test 3/4 (the Ray-based full path)")
    args = ap.parse_args()

    print(f"search_url   = {args.search_url}")
    print(f"tool_config  = {args.tool_config}")

    results = {}
    results["server_reachable"] = test_server_reachable(args.search_url)
    if not results["server_reachable"]:
        _hdr("ABORT: retrieval server unreachable")
        sys.exit(1)

    results["perform_single_search_batch"] = test_perform_single_search_batch(args.search_url)

    if not args.skip_ray:
        results["search_tool_execute"]      = test_search_tool_execute(Path(args.tool_config), args.search_url)
        results["search_tool_invalid_args"] = test_search_tool_invalid_args(Path(args.tool_config))

    _hdr("SUMMARY")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    n_fail = sum(1 for ok in results.values() if not ok)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
