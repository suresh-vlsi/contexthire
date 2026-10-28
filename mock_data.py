"""Fake candidate data with the SAME interface as crustdata.py.

All people below are invented. Use this to build and test the agent without a
Crustdata key. Shapes mimic Crustdata's search/enrich responses, so switching
to the real API later needs no pipeline changes.

Each job carries `company_industry` and `company_headcount_latest`
(`company_headcount_latest` is a real Crustdata field name; confirm the real
industry field name from an enriched record and rename here if it differs).

Some profiles are deliberately messy (gap, overlap, title jump, short tenures,
no match) to exercise red-flag logic and the verifier.
"""


class CrustdataError(Exception):
    pass


def _job(name, title, start, end=None, industry=None, hc=None):
    d = {"name": name, "title": title, "start_date": f"{start}-01T00:00:00+00:00"}
    if end:
        d["end_date"] = f"{end}-01T00:00:00+00:00"
    if industry:
        d["company_industry"] = industry
    if hc:
        d["company_headcount_latest"] = hc
    return d


def _p(pid, name, headline, title, loc, current, past, skills, school):
    return pid, {
        "basic_profile": {
            "name": name, "headline": headline, "current_title": title,
            "location": {"raw": loc},
        },
        "experience": {"employment_details": {"current": current, "past": past}},
        "education": {"schools": [school]},
        "skills": {"professional_network_skills": skills},
        "social_handles": {"professional_network_identifier": {
            "profile_url": f"https://www.linkedin.com/in/mock-{pid}"}},
    }


_PEOPLE = dict([
    _p(1001, "Aarav Mehta", "Senior Backend Engineer at PayFlow", "Senior Backend Engineer",
       "Bengaluru, Karnataka, India",
       [_job("PayFlow", "Senior Backend Engineer", "2021-03", industry="Fintech", hc=300)],
       [_job("RazorLite", "Backend Engineer", "2018-06", "2021-02", industry="Fintech", hc=60),
        _job("InfoSys", "Software Engineer", "2016-07", "2018-05", industry="IT Services", hc=300000)],
       ["Python", "Django", "PostgreSQL", "AWS", "Kafka"],
       {"school": "IIT Bombay", "degree": "B.Tech", "field_of_study": "Computer Science"}),

    _p(1002, "Diya Nair", "Staff Engineer | Payments | Ex-fintech", "Staff Software Engineer",
       "Bengaluru, Karnataka, India",
       [_job("LendWise", "Staff Software Engineer", "2022-01", industry="Fintech", hc=500)],
       [_job("PhonePay", "Senior Software Engineer", "2018-02", "2021-12", industry="Fintech", hc=5000),
        _job("Flipmart", "Software Engineer", "2014-08", "2018-01", industry="E-commerce", hc=30000)],
       ["Python", "Go", "Microservices", "Redis"],
       {"school": "NIT Trichy", "degree": "B.Tech", "field_of_study": "Electronics"}),

    # title inflation: Engineer -> Principal in 1 year; employer is accounting software
    _p(1003, "Kabir Shah", "Principal Engineer, backend", "Principal Engineer",
       "Bengaluru, Karnataka, India",
       [_job("QuickBooks India", "Principal Engineer", "2023-01", industry="Software", hc=800)],
       [_job("QuickBooks India", "Software Engineer", "2022-01", "2022-12", industry="Software", hc=800),
        _job("TechMahindra", "Associate", "2019-07", "2021-12", industry="IT Services", hc=150000)],
       ["Java", "Spring Boot", "MySQL"],
       {"school": "VIT Vellore", "degree": "B.Tech", "field_of_study": "IT"}),

    # employment gap (Mar 2020 -> Jan 2023), no fintech
    _p(1004, "Meera Iyer", "Backend developer", "Backend Developer",
       "Pune, Maharashtra, India",
       [_job("ShopNest", "Backend Developer", "2023-01", industry="E-commerce", hc=120)],
       [_job("Zomatoo", "Software Engineer", "2016-06", "2020-03", industry="Food Delivery", hc=5000)],
       ["Python", "Flask", "MongoDB"],
       {"school": "COEP Pune", "degree": "B.E.", "field_of_study": "Computer"}),

    # overlapping full-time roles
    _p(1005, "Rohan Gupta", "Full-stack and backend", "Senior Software Engineer",
       "Bengaluru, Karnataka, India",
       [_job("NeoBank Labs", "Senior Software Engineer", "2022-06", industry="Fintech", hc=45),
        _job("FreelanceHub", "Backend Lead", "2022-09", industry="Internet", hc=25)],
       [_job("Paytmm", "Software Engineer", "2017-03", "2022-05", industry="Fintech", hc=20000)],
       ["Python", "FastAPI", "AWS", "Terraform"],
       {"school": "BITS Pilani", "degree": "B.E.", "field_of_study": "CS"}),

    # many short tenures
    _p(1006, "Ishaan Rao", "Engineer, many things", "Software Engineer",
       "Hyderabad, Telangana, India",
       [_job("StartupZ", "Software Engineer", "2025-02", industry="SaaS", hc=15)],
       [_job("AppCo", "Software Engineer", "2024-03", "2025-01", industry="SaaS", hc=30),
        _job("DevShop", "Backend Engineer", "2023-04", "2024-02", industry="IT Services", hc=80),
        _job("CodeFactory", "Software Engineer", "2022-05", "2023-03", industry="IT Services", hc=200)],
       ["Python", "Node.js"],
       {"school": "JNTU", "degree": "B.Tech", "field_of_study": "CS"}),

    _p(1007, "Ananya Kulkarni", "Backend Eng @ early-stage fintech", "Backend Engineer",
       "Bengaluru, Karnataka, India",
       [_job("CredPilot (Series A)", "Backend Engineer", "2023-08", industry="Fintech", hc=40)],
       [_job("HDFC Tech", "Software Engineer", "2020-07", "2023-07",
             industry="Financial Services", hc=100000)],
       ["Python", "Django", "PostgreSQL", "Docker"],
       {"school": "IIIT Hyderabad", "degree": "B.Tech", "field_of_study": "CSE"}),

    # will come back as "no match" from enrich
    _p(1008, "Vikram Desai", "Backend Engineer", "Backend Engineer",
       "Mumbai, Maharashtra, India",
       [_job("BlueOcean Pay", "Backend Engineer", "2021-01", industry="Fintech", hc=70)],
       [],
       ["Python"],
       {"school": "Mumbai University", "degree": "B.E.", "field_of_study": "IT"}),
])

_NO_MATCH = {1008}


def variant_to_filters(v):
    """Mock: ignore details, just make sure the variant is non-empty."""
    return {"mock": True, "variant": v.name} if (v.titles or v.skills or v.locations) else None


def search_people(filters, limit: int = 25, fields=None) -> list[dict]:
    out = []
    for pid, pdata in list(_PEOPLE.items())[:limit]:
        out.append({"crustdata_person_id": pid,
                    "basic_profile": pdata["basic_profile"],
                    "social_handles": pdata["social_handles"],
                    "experience": pdata["experience"]})
    return out


def profile_url(profile: dict):
    return (profile.get("social_handles", {})
            .get("professional_network_identifier", {})
            .get("profile_url"))


def enrich_people(urls: list[str], fields=None) -> dict[str, dict | None]:
    out = {}
    for url in urls:
        pid = int(url.rsplit("-", 1)[-1])
        out[url] = None if pid in _NO_MATCH else _PEOPLE.get(pid)
    return out
