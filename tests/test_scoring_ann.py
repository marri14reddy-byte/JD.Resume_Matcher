# Minimal tests for compute_match_score and ann_shortlist_resumes
# Run with: python -m pytest -q tests/test_scoring_ann.py

import types
from pathlib import Path

# Import target module
import importlib.util

mod_path = Path(__file__).resolve().parent.parent / 'resume_matcher_rag.py'
spec = importlib.util.spec_from_file_location('resume_matcher_rag', str(mod_path))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_compute_match_score_basic():
    jd = "Company: Acme Corp\nWe need a Python engineer with 5 years of experience in AWS and Docker in the software industry. Location: Bangalore"
    rs = "Experienced Python developer (6 yrs). Skills: Python, AWS, Docker, Linux. Worked in the software and fintech domains. Based in Bengaluru."
    pct, stars, br = mod.compute_match_score(jd, rs, (0.6, 0.3, 0.1))
    assert br['skills_overlap_count'] >= 2
    assert br['industry_match'] in (0, 1)
    assert 0 <= pct <= 100


def test_ann_shortlist_resumes_handles_absence():
    # Should return None if FAISS/index is unavailable
    out = mod.ann_shortlist_resumes("sample jd", top_chunks=50, max_resumes=10)
    assert out is None or isinstance(out, list) or (isinstance(out, tuple) and isinstance(out[0], list))
