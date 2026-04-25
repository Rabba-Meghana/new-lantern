"""
Sanity checks for the prediction pipeline.
Run: python test_sanity.py
All tests must pass before submitting.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app import (
    _get_parts, _get_mods, _get_side, _years_apart,
    _is_mammography, _is_chest_xray, _is_ct_chest, _is_ct_abdomen,
    _is_mam_bilateral, _is_mam_unilateral,
    _targeted_rule, _safe_desc, _safe_date, _predict_batch,
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
check("safe_desc None",      _safe_desc({}),                        "")
check("safe_desc missing",   _safe_desc({"study_description": None}), "")
check("safe_date None",      _safe_date({}),                        "")
check("safe_desc uppercased",_safe_desc({"study_description":"ct chest"}), "CT CHEST")

print("=== Date parsing ===")
check("normal dates",     round(_years_apart("2020-01-01","2026-01-01"), 1), 6.0)
check("bad date string",  _years_apart("not-a-date","2026-01-01"),           3.0)
check("empty date",       _years_apart("", "2026-01-01"),                    3.0)

print("=== Laterality ===")
check("left",      _get_side("MRI KNEE LT WO CON"),              "left")
check("right",     _get_side("MAM DIAGNOSTIC RT WITH TOMO"),     "right")
check("bilateral", _get_side("MAM SCREEN BILAT WITH TOMO"),      "bilateral")
check("unknown",   _get_side("CT CHEST WITHOUT CONTRAST"),       "unknown")

print("=== Body part extraction ===")
check("brain",   "brain"  in _get_parts("MRI BRAIN STROKE WITHOUT CONTRAST"), True)
check("chest",   "chest"  in _get_parts("CT CHEST WITH CONTRAST"),            True)
check("no part", len(_get_parts("UNKNOWN STUDY TYPE XYZ")) == 0,              True)

print("=== Targeted rules ===")
check("mam vs cxr -> False",
      _targeted_rule("MAM SCREEN BI WITH TOMO", "XR CHEST 1V FRONTAL ONLY"), False)
check("cxr vs mam -> False",
      _targeted_rule("CHEST 2 VIEW FRONTAL & LATRL", "MAM SCREEN BI WITH TOMO"), False)
check("ct chest vs ct abd -> False",
      _targeted_rule("CT CHEST WITH CNTRST", "CT ABDOMEN PELVIS W CON"), False)
check("ct abd vs ct chest -> False",
      _targeted_rule("CT ABDOMEN PELVIS W CON", "CT CHEST WITH CNTRST"), False)
check("mam bi vs mam uni -> True",
      _targeted_rule("MAM SCREEN BI WITH TOMO", "MAM DIAGNOSTIC RT WITH TOMO"), True)
check("brain mri vs brain mri -> None (ML decides)",
      _targeted_rule("MRI BRAIN WITHOUT CONTRAST", "MRI BRAIN WITHOUT CONTRAST"), None)

print("=== Predict batch edge cases ===")
check("empty priors",  _predict_batch({}, []),  [])
check("missing study_description",
      len(_predict_batch(
          {},
          [{"study_id":"x","study_date":"2020-01-01"}]
      )) == 1, True)
check("missing study_date",
      len(_predict_batch(
          {"study_description":"CT CHEST"},
          [{"study_id":"x","study_description":"CT CHEST"}]
      )) == 1, True)
check("all predictions are bool",
      all(isinstance(v, bool) for v in _predict_batch(
          {"study_description":"MRI BRAIN","study_date":"2026-01-01"},
          [{"study_id":"1","study_description":"CT HEAD","study_date":"2020-01-01"},
           {"study_id":"2","study_description":"CT CHEST","study_date":"2021-01-01"}]
      )), True)

print(f"\n{'='*40}")
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL > 0:
    print("TESTS FAILED — do not submit")
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
