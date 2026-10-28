"""Gemini glue, built for tight free-tier quotas.

What it does to save calls:
  * disk cache: identical prompts are never paid for twice (.llm_cache/)
  * throttle: stays under GEMINI_RPM requests/minute (default 4)
  * fail fast: a *daily* quota error stops immediately instead of retrying
  * batch verify: ONE call verifies all of a candidate's claims

pip install google-genai pydantic
export GEMINI_API_KEY='...'
export GEMINI_MODEL='gemini-3.8-flash'   # whichever model your key can use
export GEMINI_RPM=4                      # set to your model's per-minute limit minus 1
"""
import hashlib
import json
import os
import re
import threading
import time
from datetime import date
from pathlib import Path

from google import genai
from google.genai import types
from pydantic import TypeAdapter

from prompts import (CLAIM_GEN_SYSTEM, PARSER_SYSTEM, VERIFIER_BATCH_SYSTEM,
                     VERIFIER_SYSTEM)
from schemas import Claim, ParsedRequest, Verdict

# Provider switch: set LLM_BASE_URL for ANY OpenAI-compatible API (Groq, OpenRouter,
# Cerebras, Mistral, Ollama...). Leave it unset to use Gemini.
BASE_URL = os.environ.get("LLM_BASE_URL")
if BASE_URL:
    from openai import OpenAI
    client = OpenAI(base_url=BASE_URL, api_key=os.environ.get("LLM_API_KEY", "none"))
    MODEL = os.environ["LLM_MODEL"]
else:
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
RPM = float(os.environ.get("LLM_RPM") or os.environ.get("GEMINI_RPM", "4"))

CACHE = Path(".llm_cache")
CACHE.mkdir(exist_ok=True)


class QuotaExhausted(RuntimeError):
    """Daily quota is gone. Retrying will not help."""


_lock = threading.Lock()
_last_call = 0.0


def _throttle():
    global _last_call
    with _lock:
        wait = (60.0 / RPM) - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()


def _generate(system: str, user: str, as_json: bool, retries: int = 3) -> str:
    key = hashlib.sha256(f"{MODEL}|{as_json}|{system}|{user}".encode()).hexdigest()
    cache_file = CACHE / f"{key}.txt"
    if cache_file.exists():
        return cache_file.read_text()

    cfg = types.GenerateContentConfig(
        system_instruction=system,
        temperature=0.2,
        response_mime_type="application/json" if as_json else "text/plain",
    )
    last = None
    for attempt in range(retries + 1):
        _throttle()
        try:
            if BASE_URL:
                r = client.chat.completions.create(
                    model=MODEL, temperature=0.2,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}])
                text = re.sub(r"<think>.*?</think>", "", r.choices[0].message.content or "",
                              flags=re.S).strip()
            else:
                text = client.models.generate_content(model=MODEL, contents=user, config=cfg).text
            if text:
                cache_file.write_text(text)
                return text
            last = "empty response"
        except Exception as e:
            msg = str(e)
            last = msg
            if "PerDay" in msg or "per day" in msg.lower():
                raise QuotaExhausted(
                    f"Daily quota exhausted for {MODEL}. Switch GEMINI_MODEL, use another "
                    "project/key, enable billing, or wait for the reset.") from e
            m = re.search(r"(?:retry|try again) in ([\d.]+)\s?s", msg)
            if "429" in msg and m:               # per-minute limit: wait as told
                time.sleep(float(m.group(1)) + 1)
                continue
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Gemini call failed after retries: {last}")


def call_json(system: str, user: str, retries: int = 2):
    last_err = None
    for _ in range(retries + 1):
        text = _generate(system, user, as_json=True)
        text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            last_err = e
            user += f"\n\nYour last reply was not valid JSON ({e}). Return JSON only."
    raise ValueError(f"LLM never returned valid JSON: {last_err}")


def generate_text(system: str, user: str) -> str:
    return _generate(system, user, as_json=False).strip()


# ---- Stage 1: parse ----
def parse_request(raw_request: str) -> ParsedRequest:
    return ParsedRequest.model_validate(call_json(PARSER_SYSTEM, raw_request))


# ---- Stage 4a: generate claims for one candidate (1 call) ----
def generate_claims(candidate: dict, criteria: list, companies: list) -> list[Claim]:
    user = json.dumps({
        "criteria": [c.model_dump(mode="json") for c in criteria],
        "candidate_record": candidate,
        "company_records": companies,
        "today": date.today().isoformat(),
    })
    return TypeAdapter(list[Claim]).validate_python(call_json(CLAIM_GEN_SYSTEM, user))


# ---- Stage 4b: verify ----
def _quote_check(claim: Claim, records: dict[str, dict]) -> bool:
    """Programmatic check: every quoted evidence value must exist in its record."""
    for ev in claim.evidence:
        raw = json.dumps(records.get(ev.record_id, {}), ensure_ascii=False)
        if ev.value not in raw:
            claim.verdict = Verdict.UNSUPPORTED
            claim.verifier_note = f"Quoted value not found in record {ev.record_id}."
            return False
    return True


def verify_claims(claims: list[Claim], records: dict[str, dict]) -> list[Claim]:
    """Batch verify: programmatic check first, then ONE LLM call for the rest."""
    pending = [c for c in claims if _quote_check(c, records)]
    if not pending:
        return claims
    cited_ids = {ev.record_id for c in pending for ev in c.evidence}
    user = json.dumps({
        "today": date.today().isoformat(),
        "claims": [c.model_dump(mode="json") for c in pending],
        "sources": {rid: records[rid] for rid in cited_ids if rid in records},
    })
    out = call_json(VERIFIER_BATCH_SYSTEM, user)
    by_id = {str(o.get("id")): o for o in out if isinstance(o, dict)}
    for c in pending:
        o = by_id.get(str(c.id))
        if not o:
            c.verdict = Verdict.UNSUPPORTED
            c.verifier_note = "Verifier returned no verdict for this claim."
            continue
        try:
            c.verdict = Verdict(o["verdict"])
        except (KeyError, ValueError):
            c.verdict = Verdict.UNSUPPORTED
        c.verifier_note = o.get("note", "")
        c.corrected_text = o.get("corrected_text")
    return claims


def verify_all(claims: list[Claim], records: dict[str, dict]) -> list[Claim]:
    return verify_claims(claims, records)


def renderable(claims: list[Claim]) -> list[Claim]:
    return [c for c in claims if c.verified]


if __name__ == "__main__":
    parsed = parse_request("need a senior backend eng, BLR, ex-fintech, startup vibes")
    print(parsed.model_dump_json(indent=2))
