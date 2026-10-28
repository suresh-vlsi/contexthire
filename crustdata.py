"""Thin Crustdata client for Person Search + Person Enrich.

Built from the public docs (docs.crustdata.com, API version 2025-11-01):
  POST /person/search  : body {filters, fields, limit, cursor?}  -> {profiles, next_cursor, total_count}
                         rate limit 30 req/min
  POST /person/enrich  : body {professional_network_profile_urls[<=25], fields}
                         -> [{matched_on, match_status, matches:[{confidence_score, person_data}]}]
                         rate limit 15 req/min

pip install requests
export CRUSTDATA_API_KEY=...
"""
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import requests

BASE = "https://api.crustdata.com"
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)


class CrustdataError(Exception):
    pass


class _Throttle:
    """Keeps requests under the per-minute limit for one endpoint."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            delta = time.time() - self._last
            if delta < self.min_interval:
                time.sleep(self.min_interval - delta)
            self._last = time.time()


_throttles = {
    "/person/search": _Throttle(2.1),   # 30/min
    "/person/enrich": _Throttle(4.2),   # 15/min
}


def _headers():
    key = os.environ.get("CRUSTDATA_API_KEY")
    if not key:
        raise CrustdataError("Set CRUSTDATA_API_KEY")
    return {
        "authorization": f"Bearer {key}",
        "x-api-version": "2025-11-01",
        "content-type": "application/json",
    }


def _post(path: str, body: dict, retries: int = 4):
    """POST with disk cache, throttle, and retry on 429/5xx."""
    key = hashlib.sha256((path + json.dumps(body, sort_keys=True)).encode()).hexdigest()
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    last = None
    for attempt in range(retries + 1):
        _throttles[path].wait()
        try:
            r = requests.post(BASE + path, headers=_headers(), json=body, timeout=60)
        except requests.RequestException as e:
            last = str(e)
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            data = r.json()
            cache_file.write_text(json.dumps(data))
            return data
        if r.status_code == 429 or r.status_code >= 500:
            last = f"{r.status_code}: {r.text[:200]}"
            time.sleep(float(r.headers.get("retry-after", 2 ** (attempt + 1))))
            continue
        # 400 / 401 / 403: our fault, do not retry
        raise CrustdataError(f"{path} {r.status_code}: {r.text[:500]}")
    raise CrustdataError(f"{path} failed after retries: {last}")


# ---------------------------------------------------------------- search

SEARCH_FIELDS = ["crustdata_person_id", "basic_profile", "social_handles", "experience"]


def _contains(field: str, value: str) -> dict:
    # "(.)" = every word in value appears, any order, case-insensitive
    return {"field": field, "type": "(.)", "value": value}


def variant_to_filters(v) -> dict | None:
    """Turn a SearchVariant (schemas.py) into a Crustdata filter tree."""
    conds = []
    if v.titles:
        conds.append({"op": "or", "conditions": [
            _contains("experience.employment_details.current.title", t) for t in v.titles]})
    if v.locations:
        conds.append({"op": "or", "conditions": [
            _contains("basic_profile.location.full_location", loc) for loc in v.locations]})
    for skill in v.skills[:3]:   # more than 3 AND-ed skills usually returns nothing
        conds.append(_contains("skills.professional_network_skills", skill))
    if v.min_years_experience:
        conds.append({"field": "years_of_experience_raw", "type": "=>",
                      "value": v.min_years_experience})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"op": "and", "conditions": conds}


def search_people(filters: dict, limit: int = 25, fields=None) -> list[dict]:
    body = {"filters": filters, "fields": fields or SEARCH_FIELDS, "limit": limit}
    return _post("/person/search", body).get("profiles", [])


def profile_url(profile: dict) -> str | None:
    return (profile.get("social_handles", {})
            .get("professional_network_identifier", {})
            .get("profile_url"))


# ---------------------------------------------------------------- enrich

ENRICH_FIELDS = ["basic_profile", "experience", "education", "skills", "social_handles"]


def enrich_people(urls: list[str], fields=None) -> dict[str, dict | None]:
    """Returns {input_url: person_data or None}. Batches of 25."""
    out: dict[str, dict | None] = {}
    for i in range(0, len(urls), 25):
        chunk = urls[i:i + 25]
        body = {"professional_network_profile_urls": chunk,
                "fields": fields or ENRICH_FIELDS}
        for entry in _post("/person/enrich", body):
            url = entry.get("matched_on")
            matches = entry.get("matches") or []
            # take the best match; skip redacted / not_found / low confidence
            best = max(matches, key=lambda m: m.get("confidence_score", 0), default=None)
            out[url] = best["person_data"] if best and best.get("confidence_score", 0) >= 0.8 else None
    return out
