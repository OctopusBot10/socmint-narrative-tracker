"""ClaimBuster API v2 client.

The score endpoint rates how check-worthy a sentence is. 0.0 is opinion or
non-factual wording. 1.0 is a highly check-worthy factual claim. The score is
not a true/false verdict.

Register a key in the ClaimBuster portal and set CLAIMBUSTER_API_KEY in .env.
"""

from __future__ import annotations

from urllib.parse import quote

import requests

from config import env

API_ROOT = "https://idir.uta.edu/api/v2"
SCORE_TEXT_URL = f"{API_ROOT}/score/text/"
SCORE_SENTENCES_URL = f"{API_ROOT}/score/text/sentences/"
REQUEST_TIMEOUT_SECONDS = 5

_score_cache: dict[str, float] = {}


def has_api_key() -> bool:
    return bool(env("CLAIMBUSTER_API_KEY"))


def get_claim_score(text: str) -> float:
    """Return checkworthiness as a percentage from 0.0 to 100.0.

    An empty key, a blank sentence, or a failed request returns 0.0.
    """
    claim = " ".join(text.split())
    if not claim:
        return 0.0
    key = env("CLAIMBUSTER_API_KEY")
    if not key:
        return 0.0
    if claim in _score_cache:
        return _score_cache[claim]
    try:
        response = requests.post(
            SCORE_TEXT_URL,
            json={"input_text": claim},
            headers={"x-api-key": key},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            print(f"ClaimBuster scoring error: HTTP {response.status_code}")
            return 0.0
        data = response.json()
        results = data.get("results", [])
        if not results:
            return 0.0
        score = round(float(results[0].get("score", 0.0)) * 100, 1)
    except Exception as exc:
        print(f"ClaimBuster scoring error: {exc}")
        return 0.0
    _score_cache[claim] = score
    return score


def score_sentences(text: str) -> list[dict]:
    """Score each sentence. Returns ClaimBuster's JSON list, or an empty list."""
    body = _post(SCORE_SENTENCES_URL, text)
    if isinstance(body, list):
        return body
    if isinstance(body, dict):
        results = body.get("results")
        if isinstance(results, list):
            return results
    return []


def query_knowledge_bases(claim: str) -> dict:
    """Look up a claim in ClaimBuster knowledge bases."""
    return _get(f"/query/knowledge_bases/{_path_claim(claim)}")


def query_fact_matcher(claim: str) -> dict:
    """Match a claim against ClaimBuster's fact-check database."""
    return _get(f"/query/fact_matcher/{_path_claim(claim)}")


def claim_similarity(claim_a: str, claim_b: str) -> dict:
    """Score how similar two claims are."""
    return _get(
        "/claim_similarity/simple_similarity/score/"
        f"claim_a/{_path_claim(claim_a)}/claim_b/{_path_claim(claim_b)}"
    )


def _post(url: str, text: str) -> dict | list:
    key = env("CLAIMBUSTER_API_KEY")
    if not key or not text.strip():
        return {}
    try:
        response = requests.post(
            url,
            json={"input_text": " ".join(text.split())},
            headers={"x-api-key": key},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            print(f"ClaimBuster scoring error: HTTP {response.status_code}")
            return {}
        return response.json()
    except Exception as exc:
        print(f"ClaimBuster scoring error: {exc}")
        return {}


def _get(path: str) -> dict:
    key = env("CLAIMBUSTER_API_KEY")
    if not key:
        return {}
    try:
        response = requests.get(
            f"{API_ROOT}{path}",
            headers={"x-api-key": key},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            print(f"ClaimBuster scoring error: HTTP {response.status_code}")
            return {}
        body = response.json()
        return body if isinstance(body, dict) else {"results": body}
    except Exception as exc:
        print(f"ClaimBuster scoring error: {exc}")
        return {}


def _path_claim(claim: str) -> str:
    compact = " ".join(claim.split())[:400]
    return quote(compact, safe="")
