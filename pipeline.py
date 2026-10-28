"""End-to-end recruiter agent.

Usage:
    python pipeline.py "need a senior backend eng in Bangalore, ex-fintech" [--top 10] [--out report.md]

Flow: parse -> search (strict -> relaxed -> adjacent until enough) -> enrich
      -> claims -> verify -> score -> markdown report. Never contacts anyone.
"""
import argparse
import re
from datetime import date
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import os

# Real Crustdata when a key is set, otherwise the built-in mock candidates.
USE_MOCK = os.environ.get("USE_MOCK_DATA") == "1" or not os.environ.get("CRUSTDATA_API_KEY")
if USE_MOCK:
    from mock_data import (CrustdataError, enrich_people, profile_url,
                           search_people, variant_to_filters)
else:
    from crustdata import (CrustdataError, enrich_people, profile_url,
                           search_people, variant_to_filters)
from derived import compute_derived
from llm import QuotaExhausted, generate_claims, generate_text, parse_request, verify_all
from schemas import (CandidateReport, Claim, CriterionScore, ParsedRequest,
                     Priority, Verdict)

MIN_CANDIDATES = 12      # keep loosening search until we have this many
SEARCH_LIMIT = 25        # per variant
WEIGHTS = {Priority.MUST: 3.0, Priority.NICE: 1.0}
CREDIT = {"yes": 1.0, "partial": 0.5, "no": 0.0, "unknown": 0.0}


def log(msg: str):
    print(f"[agent] {msg}", file=sys.stderr)


# ------------------------------------------------------------ stage 2: search

def gather_candidates(parsed: ParsedRequest) -> dict[str, dict]:
    """Returns {rid: {"url":..., "variant":...}} deduped by crustdata_person_id."""
    found: dict[str, dict] = {}
    for variant in parsed.search_variants:
        filters = variant_to_filters(variant)
        if not filters:
            log(f"variant '{variant.name}' has no usable filters, skipping")
            continue
        try:
            profiles = search_people(filters, limit=SEARCH_LIMIT)
        except CrustdataError as e:
            log(f"variant '{variant.name}' failed: {e}")
            continue
        added = 0
        for p in profiles:
            rid, url = str(p.get("crustdata_person_id")), profile_url(p)
            if rid != "None" and url and rid not in found:
                found[rid] = {"url": url, "variant": variant.name, "profile": p}
                added += 1
        log(f"variant '{variant.name}': {len(profiles)} results, {added} new")
        if len(found) >= MIN_CANDIDATES:
            break
    return found


# ------------------------------------------------------------ pre-ranking (no LLM, no credits)

def _blob(profile: dict) -> str:
    bp = profile.get("basic_profile", {})
    parts = [bp.get("headline", ""), bp.get("current_title", ""),
             (bp.get("location") or {}).get("raw", "")]
    ed = profile.get("experience", {}).get("employment_details", {})
    for j in (ed.get("current") or []) + (ed.get("past") or []):
        parts += [j.get("name", ""), j.get("title", "")]
    return " ".join(parts).lower()


def _frac(phrase: str, blob: str) -> float:
    """Fraction of the phrase's words present in the blob."""
    words = [w for w in re.findall(r"\w+", phrase.lower()) if len(w) > 2]
    return sum(w in blob for w in words) / len(words) if words else 0.0


def prerank(profile: dict, parsed: ParsedRequest, variant_name: str) -> float:
    """Cheap relevance score from search results, so LLM calls and enrich
    credits go to the most promising candidates instead of the first few."""
    strict = next((v for v in parsed.search_variants if v.name == "strict"),
                  parsed.search_variants[0])
    blob = _blob(profile)
    score = 3 * max((_frac(t, blob) for t in strict.titles), default=0)
    score += 2 * max((_frac(l, blob) for l in strict.locations), default=0)
    score += 0.5 * sum(_frac(k, blob) for k in strict.skills)
    score += 1 if variant_name == "strict" else 0
    return score


# ------------------------------------------------------------ stage 3-4: per candidate

def process_candidate(rid: str, person_data: dict, parsed: ParsedRequest):
    """Generate claims and verify them. Returns (claims, records)."""
    record = {"record_id": rid, **person_data,
              "derived": compute_derived(person_data, date.today())}
    records = {rid: record}
    claims = generate_claims(record, parsed.criteria, companies=[])
    for c in claims:
        c.candidate_id = rid
    verified = verify_all(claims, records)
    return verified


# ------------------------------------------------------------ stage 5: score + render

def score_candidate(rid: str, name: str, claims: list[Claim], parsed: ParsedRequest) -> CandidateReport:
    scores, earned, total = [], 0.0, 0.0
    must_failed = False
    for crit in parsed.criteria:
        mine = [c for c in claims if c.criterion_id == crit.id]
        if any(c.verdict == Verdict.SUPPORTED for c in mine):
            met = "yes"
        elif any(c.verdict == Verdict.PARTIAL for c in mine):
            met = "partial"
        elif any(c.verdict == Verdict.CONTRADICTED for c in mine):
            met = "no"
        else:
            met = "unknown"
        if met == "no" and crit.priority == Priority.MUST:
            must_failed = True
        w = WEIGHTS[crit.priority]
        earned += w * CREDIT[met]
        total += w
        scores.append(CriterionScore(criterion_id=crit.id, met=met,
                                     claim_ids=[c.id for c in mine if c.verified]))
    pct = 100 * earned / total if total else 0
    if must_failed:
        pct *= 0.5
    n_flags = sum(1 for c in claims if c.criterion_id is None and c.verified)
    pct = max(0.0, pct - min(15, 5 * n_flags))      # -5 per red flag, max -15
    return CandidateReport(candidate_id=rid, name=name, score=round(pct, 1),
                           criterion_scores=scores, claims=claims)


def draft_outreach(report: CandidateReport, role: str) -> str:
    """DRAFT ONLY. Uses verified claims only. Never sent anywhere."""
    facts = [c.corrected_text or c.text for c in report.claims if c.verified][:3]
    return generate_text(
        "Write a short, warm recruiter outreach message (under 90 words). "
        "Use ONLY the facts given. No flattery beyond them. No invented details.",
        f"Role: {role}\nCandidate: {report.name}\nFacts:\n- " + "\n- ".join(facts))


def render(parsed: ParsedRequest, reports: list[CandidateReport], meta: dict) -> str:
    crit = {c.id: c for c in parsed.criteria}
    L = [f"# Shortlist: {parsed.role_title}", ""]
    if parsed.assumptions:
        L += ["**Assumptions made:**"] + [f"- {a}" for a in parsed.assumptions] + [""]
    if parsed.conflicts:
        L += ["**Conflicts in the request:**"] + [f"- {a}" for a in parsed.conflicts] + [""]
    L += [f"_Searched {meta['found']} candidates, enriched {meta['enriched']}, analysed the best "
          f"{meta['analysed']}, generated {meta['claims']} claims, "
          f"**{meta['rejected']} rejected by the verifier**. Each red flag lowers the score "
          f"by 5 points (max 15). Experience figures are computed by code from record dates._", ""]
    for i, r in enumerate(reports, 1):
        L += [f"## {i}. {r.name} ({r.score}/100)", ""]
        L += ["| Criterion | Priority | Met |", "|---|---|---|"]
        for s in r.criterion_scores:
            c = crit[s.criterion_id]
            L.append(f"| {c.text} | {c.priority.value} | {s.met} |")
        L += ["", "**Evidence**"]
        for c in r.claims:
            if c.verified:
                ev = "; ".join(f"`{e.record_id}:{e.field_path}` = {e.value!r}" for e in c.evidence)
                changed = bool(c.corrected_text) and c.corrected_text.strip() != c.text.strip()
                tag = (" _(corrected)_" if changed
                       else " _(partially supported)_" if c.verdict == Verdict.PARTIAL else "")
                L.append(f"- {c.corrected_text or c.text}{tag}  \n  <sub>{ev}</sub>")
        flags = [c for c in r.claims if c.criterion_id is None and c.verified]
        if flags:
            L += ["", "**Red flags**"] + [f"- {c.corrected_text or c.text}" for c in flags]
        rej = [c for c in r.claims if not c.verified]
        if rej:
            L += ["", "<details><summary>Rejected claims</summary>", ""]
            L += [f"- ~~{c.text}~~ ({c.verdict.value}: {c.verifier_note})" for c in rej]
            L += ["", "</details>"]
        if r.outreach_draft:
            L += ["", "**Outreach draft (not sent)**", "", f"> {r.outreach_draft}"]
        L.append("")
    return "\n".join(L)


# ------------------------------------------------------------ main

def run(request: str, top: int = 10, out: str = "report.md", outreach: bool = False):
    log("data source: " + ("MOCK candidates (no Crustdata key)" if USE_MOCK else "Crustdata API"))
    log("parsing request")
    parsed = parse_request(request)
    log(f"{len(parsed.criteria)} criteria, {len(parsed.search_variants)} variants")

    found = gather_candidates(parsed)
    if not found:
        msg = "No candidates found even after loosening filters. Try a broader request."
        log(msg)
        open(out, "w").write(f"# {parsed.role_title}\n\n{msg}\n")
        return

    # rank by search-result relevance, enrich the best (a couple extra: some will not match)
    ranked = sorted(found.values(),
                    key=lambda v: prerank(v["profile"], parsed, v["variant"]), reverse=True)
    urls = [v["url"] for v in ranked[: top + 2]]
    log(f"pre-ranked {len(found)} candidates, enriching the best {len(urls)}")
    enriched = enrich_people(urls)
    url_to_rid = {v["url"]: rid for rid, v in found.items()}

    people = {}
    for url, pdata in enriched.items():
        rid = url_to_rid.get(url)
        if rid and pdata:
            people[rid] = pdata
    log(f"{len(people)} profiles enriched successfully")
    n_enriched = len(people)
    people = dict(list(people.items())[:top])      # LLM quota is tight: process only `top`
    log(f"processing {len(people)} candidates")

    def work(item):
        rid, pdata = item
        try:
            claims = process_candidate(rid, pdata, parsed)
            name = pdata.get("basic_profile", {}).get("name", rid)
            return score_candidate(rid, name, claims, parsed)
        except QuotaExhausted as e:
            log(f"STOPPING candidate {rid}: {e}")
            return None
        except Exception as e:           # one bad candidate must not kill the run
            log(f"candidate {rid} failed: {e}")
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        reports = [r for r in pool.map(work, people.items()) if r]

    if not reports:
        msg = "No candidate could be processed (LLM errors or quota). See the log."
        log(msg)
        open(out, "w").write(f"# {parsed.role_title}\n\n{msg}\n")
        return
    reports.sort(key=lambda r: r.score, reverse=True)
    reports = reports[:top]
    if outreach:
        for r in reports[:3]:
            r.outreach_draft = draft_outreach(r, parsed.role_title)

    all_claims = [c for r in reports for c in r.claims]
    meta = {"found": len(found), "enriched": n_enriched, "analysed": len(reports), "claims": len(all_claims),
            "rejected": sum(1 for c in all_claims if not c.verified)}
    open(out, "w").write(render(parsed, reports, meta))
    log(f"wrote {out}  {json.dumps(meta)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("request")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", default="report.md")
    ap.add_argument("--outreach", action="store_true")
    a = ap.parse_args()
    run(a.request, a.top, a.out, a.outreach)
