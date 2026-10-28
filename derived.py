"""Deterministic facts computed from a person record's dates.

LLMs (especially small free ones) get date arithmetic wrong, so we do it here
and put the result in the record under `derived`. Claims then cite those
fields, and the verifier treats them as trusted.

Everything here is computed from the record's own start/end dates. Red-flag
rules (thresholds are easy to tune):
  gap            : more than ~6 months with no role
  overlap        : two roles at DIFFERENT companies overlapping > 90 days
  short_tenures  : 3+ ended roles shorter than 12 months
  title_jump     : same company, new title 3+ seniority levels higher
"""
import re
from datetime import date, datetime

GAP_DAYS = 183
OVERLAP_DAYS = 90
SHORT_TENURE_DAYS = 365
JUMP_LEVELS = 3

_RANKS = [
    (("chief", "cto", "ceo", "vp", "vice president"), 8),
    (("director", "head"), 7),
    (("principal", "distinguished", "fellow"), 6),
    (("staff",), 5),
    (("lead", "manager"), 4),
    (("senior", "sr"), 3),
    (("junior", "jr", "associate", "intern", "trainee"), 1),
]


def _rank(title: str) -> int:
    t = (title or "").lower()
    for words, rank in _RANKS:
        if any(re.search(rf"\b{re.escape(w)}\b", t) for w in words):
            return rank
    return 2


def _d(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _fmt(d: date) -> str:
    return d.strftime("%Y-%m")


def _months(days: int) -> int:
    return round(days / 30.44)


def compute_derived(person: dict, today: date) -> dict:
    ed = person.get("experience", {}).get("employment_details", {})
    roles = []
    for kind in ("current", "past"):
        for j in ed.get(kind) or []:
            s = _d(j.get("start_date"))
            e = _d(j.get("end_date")) if j.get("end_date") else (today if kind == "current" else None)
            if not s or not e or e < s:
                continue
            roles.append({"name": j.get("name") or "", "title": j.get("title") or "",
                          "s": s, "e": e, "ongoing": kind == "current"})
    roles.sort(key=lambda r: r["s"])

    out = {"as_of": today.isoformat(),
           "note": "Computed by code from this record's start/end dates."}
    if not roles:
        out.update(total_experience="unknown (no dated roles)", role_durations=[], red_flags=[])
        return out

    # merge overlapping periods so experience is not double counted
    merged = []
    for r in roles:
        if merged and r["s"] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], r["e"])
        else:
            merged.append([r["s"], r["e"]])
    total_days = sum((e - s).days for s, e in merged)
    out["total_experience"] = (f"{total_days / 365.25:.1f} years across {len(roles)} dated roles, "
                               f"measured to {today.isoformat()}, overlapping periods counted once")
    out["first_role_started"] = _fmt(roles[0]["s"])
    out["role_durations"] = [
        f"{r['name']}: {r['title']}, {_fmt(r['s'])} to "
        f"{'present' if r['ongoing'] else _fmt(r['e'])}, {(r['e'] - r['s']).days / 365.25:.1f} years"
        for r in roles]

    flags = []

    for (s1, e1), (s2, e2) in zip(merged, merged[1:]):
        gap = (s2 - e1).days
        if gap > GAP_DAYS:
            flags.append({"kind": "gap",
                          "text": f"Employment gap of about {_months(gap)} months between "
                                  f"{_fmt(e1)} and {_fmt(s2)}"})

    for i, a in enumerate(roles):
        for b in roles[i + 1:]:
            if a["name"].lower() == b["name"].lower():
                continue
            start, end = max(a["s"], b["s"]), min(a["e"], b["e"])
            if (end - start).days > OVERLAP_DAYS:
                flags.append({"kind": "overlap",
                              "text": f"Overlapping roles at {a['name']} ({a['title']}) and "
                                      f"{b['name']} ({b['title']}) from {_fmt(start)} to "
                                      f"{'present' if end == today else _fmt(end)}"})

    short = [r for r in roles if not r["ongoing"] and (r["e"] - r["s"]).days < SHORT_TENURE_DAYS]
    if len(short) >= 3:
        flags.append({"kind": "short_tenures",
                      "text": f"{len(short)} roles shorter than 12 months: "
                              + "; ".join(f"{r['name']} ({_fmt(r['s'])} to {_fmt(r['e'])})" for r in short)})

    for i, a in enumerate(roles):
        for b in roles[i + 1:]:
            if (a["name"].lower() == b["name"].lower() and 0 <= (b["s"] - a["e"]).days <= 92
                    and _rank(b["title"]) - _rank(a["title"]) >= JUMP_LEVELS):
                flags.append({"kind": "title_jump",
                              "text": f"Title jump at {a['name']}: {a['title']} to {b['title']} "
                                      f"within {_months((b['s'] - a['s']).days)} months"})

    out["red_flags"] = flags
    return out


if __name__ == "__main__":      # quick self-test on the mock people
    import json
    import mock_data
    for pid, pdata in mock_data._PEOPLE.items():
        d = compute_derived(pdata, date(2026, 10, 28))
        print(pid, pdata["basic_profile"]["name"], "|", d["total_experience"][:22])
        for f in d["red_flags"]:
            print("    ", f["kind"], "-", f["text"])
