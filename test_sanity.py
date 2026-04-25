"""
Sanity checks for the prediction pipeline.
Run: python test_sanity.py
All 32 tests must pass before submitting.

Imports from features.py (shared module) and app.py (inference logic).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from features import (
    get_parts, get_mods, get_side, years_apart, build_features,
    safe_desc, safe_date,
    is_mammography, is_chest_xray, is_ct_chest, is_ct_abdomen,
    is_mam_bilateral, is_mam_unilateral, targeted_rule,
)
from app import _cache_key, _predict_batch

PASS = 0; FAIL = 0

def check(name, got, expected):
    global PASS, FAIL
    if got == expected:
        print(f"  PASS  {name}")
        PASS += 1
    else:
        print(f"  FAIL  {name}  got={got!r}  expected={expected!r}")
        FAIL += 1

print("=== Input safety ===")
check("safe_desc None",        safe_desc({}),                           "")
check("safe_desc missing key", safe_desc({"study_description": None}),  "")
check("safe_date None",        safe_date({}),                           "")
check("safe_desc uppercased",  safe_desc({"study_description":"ct chest"}), "CT CHEST")

print("=== Cache key correctness ===")
k1 = _cache_key("CT CHEST", "2026-01-01", "CT CHEST", "2020-01-01")
k2 = _cache_key("CT CHEST", "2025-01-01", "CT CHEST", "2020-01-01")
check("different cur_date -> different key", k1 != k2, True)
check("identical inputs -> same key", k1 == _cache_key("CT CHEST","2026-01-01","CT CHEST","2020-01-01"), True)

print("=== Date parsing ===")
check("normal dates",    round(years_apart("2020-01-01","2026-01-01"), 1), 6.0)
check("bad date string", years_apart("not-a-date","2026-01-01"),           3.0)
check("empty date",      years_apart("","2026-01-01"),                     3.0)

print("=== Laterality ===")
check("left",      get_side("MRI KNEE LT WO CON"),          "left")
check("right",     get_side("MAM DIAGNOSTIC RT WITH TOMO"), "right")
check("bilateral", get_side("MAM SCREEN BILAT WITH TOMO"),  "bilateral")
check("unknown",   get_side("CT CHEST WITHOUT CONTRAST"),   "unknown")

print("=== Body part extraction ===")
check("brain in MRI BRAIN", "brain" in get_parts("MRI BRAIN STROKE WITHOUT CONTRAST"), True)
check("chest in CT CHEST",  "chest" in get_parts("CT CHEST WITH CONTRAST"),            True)
check("no part for unknown", len(get_parts("UNKNOWN STUDY TYPE XYZ")) == 0,            True)

print("=== Modality classifiers ===")
check("is_mammography mam",      is_mammography("MAM SCREEN BI WITH TOMO"),         True)
check("is_mammography breast",   is_mammography("US BREAST LT"),                    True)
check("is_mammography false",    is_mammography("CT CHEST"),                         False)
check("is_chest_xray",           is_chest_xray("CHEST 2 VIEW FRONTAL & LATRL"),     True)
check("is_chest_xray not mam",   is_chest_xray("MAM SCREEN BI"),                    False)
check("is_ct_chest",             is_ct_chest("CT CHEST WITH CNTRST"),               True)
check("is_ct_abdomen",           is_ct_abdomen("CT ABDOMEN PELVIS W CON"),          True)
check("is_mam_bilateral screen", is_mam_bilateral("MAM SCREEN BI WITH TOMO"),       True)
check("is_mam_unilateral rt",    is_mam_unilateral("MAM DIAGNOSTIC RT WITH TOMO"),  True)
check("is_mam_unilateral bi",    is_mam_unilateral("MAM SCREEN BILAT"),             False)

print("=== Targeted rules ===")
check("mam vs cxr -> False",
      targeted_rule("MAM SCREEN BI WITH TOMO", "CHEST 2 VIEW FRONTAL & LATRL"), False)
check("ct chest vs ct abd -> False",
      targeted_rule("CT CHEST WITH CNTRST", "CT ABDOMEN PELVIS W CON"),         False)
check("mam bi vs mam uni -> True",
      targeted_rule("MAM SCREEN BI WITH TOMO", "MAM DIAGNOSTIC RT WITH TOMO"),  True)
check("brain mri vs brain mri -> None",
      targeted_rule("MRI BRAIN WITHOUT CONTRAST", "MRI BRAIN WITHOUT CONTRAST"), None)

print("=== Predict batch edge cases ===")
check("empty priors", _predict_batch({}, []), [])
check("all predictions are bool",
      all(isinstance(v, bool) for v in _predict_batch(
          {"study_description":"MRI BRAIN","study_date":"2026-01-01"},
          [{"study_id":"1","study_description":"CT HEAD","study_date":"2020-01-01"},
           {"study_id":"2","study_description":"CT CHEST","study_date":"2021-01-01"}]
      )), True)

print(f"\n{'='*40}")
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL:
    print("TESTS FAILED — do not submit"); sys.exit(1)
else:
    print("ALL TESTS PASSED")
