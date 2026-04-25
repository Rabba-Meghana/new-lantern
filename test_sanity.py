"""
Sanity checks for the prediction pipeline.
Run: python test_sanity.py
All 24 tests must pass before submitting.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from app import (
    _get_parts, _get_mods, _get_side, _years_apart,
    is_mammography, is_chest_xray, is_ct_chest, is_ct_abdomen,
    is_mam_bilateral, is_mam_unilateral,
    _targeted_rule, _safe_desc, _safe_date, _cache_key, _predict_batch,
)

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
check("safe_desc None",        _safe_desc({}),                           "")
check("safe_desc missing key", _safe_desc({"study_description": None}),  "")
check("safe_date None",        _safe_date({}),                           "")
check("safe_desc uppercased",  _safe_desc({"study_description":"ct chest"}), "CT CHEST")

print("=== Cache key includes cur_date ===")
k1 = _cache_key("CT CHEST", "2026-01-01", "CT CHEST", "2020-01-01")
k2 = _cache_key("CT CHEST", "2025-01-01", "CT CHEST", "2020-01-01")
check("different cur_date produces different key", k1 != k2, True)
check("same inputs produce same key", k1 == _cache_key("CT CHEST", "2026-01-01", "CT CHEST", "2020-01-01"), True)

print("=== Date parsing ===")
check("normal dates",    round(_years_apart("2020-01-01","2026-01-01"), 1), 6.0)
check("bad date string", _years_apart("not-a-date","2026-01-01"),           3.0)
check("empty date",      _years_apart("","2026-01-01"),                     3.0)

print("=== Laterality ===")
check("left",      _get_side("MRI KNEE LT WO CON"),          "left")
check("right",     _get_side("MAM DIAGNOSTIC RT WITH TOMO"), "right")
check("bilateral", _get_side("MAM SCREEN BILAT WITH TOMO"),  "bilateral")
check("unknown",   _get_side("CT CHEST WITHOUT CONTRAST"),   "unknown")

print("=== Body part extraction ===")
check("brain in MRI BRAIN", "brain" in _get_parts("MRI BRAIN STROKE WITHOUT CONTRAST"), True)
check("chest in CT CHEST",  "chest" in _get_parts("CT CHEST WITH CONTRAST"),            True)
check("no part for unknown", len(_get_parts("UNKNOWN STUDY TYPE XYZ")) == 0,            True)

print("=== Modality classifiers ===")
check("is_mammography mam",     is_mammography("MAM SCREEN BI WITH TOMO"),       True)
check("is_mammography breast",  is_mammography("US BREAST LT"),                  True)
check("is_mammography false",   is_mammography("CT CHEST"),                       False)
check("is_chest_xray",          is_chest_xray("CHEST 2 VIEW FRONTAL & LATRL"),   True)
check("is_chest_xray not mam",  is_chest_xray("MAM SCREEN BI"),                  False)
check("is_ct_chest",            is_ct_chest("CT CHEST WITH CNTRST"),             True)
check("is_ct_abdomen",          is_ct_abdomen("CT ABDOMEN PELVIS W CON"),        True)
check("is_mam_bilateral screen",is_mam_bilateral("MAM SCREEN BI WITH TOMO"),    True)
check("is_mam_unilateral rt",   is_mam_unilateral("MAM DIAGNOSTIC RT WITH TOMO"), True)
check("is_mam_unilateral bi",   is_mam_unilateral("MAM SCREEN BILAT"),           False)

print("=== Targeted rules ===")
check("mam vs cxr -> False",
      _targeted_rule("MAM SCREEN BI WITH TOMO", "CHEST 2 VIEW FRONTAL & LATRL"), False)
check("ct chest vs ct abd -> False",
      _targeted_rule("CT CHEST WITH CNTRST", "CT ABDOMEN PELVIS W CON"),         False)
check("mam bi vs mam uni -> True",
      _targeted_rule("MAM SCREEN BI WITH TOMO", "MAM DIAGNOSTIC RT WITH TOMO"), True)
check("brain mri vs brain mri -> None",
      _targeted_rule("MRI BRAIN WITHOUT CONTRAST", "MRI BRAIN WITHOUT CONTRAST"), None)

print("=== Predict batch edge cases ===")
check("empty priors",  _predict_batch({}, []), [])
check("missing description returns bool",
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
