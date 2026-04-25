"""
Shared feature extraction, rule logic, and constants.
Imported by both app.py (inference) and train.py (training).
This module is the single source of truth — no duplication.
"""

import json
import os
from datetime import datetime

# ── Load config ────────────────────────────────────────────────────────────────
_cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
with open(_cfg_path) as _f:
    CFG = json.load(_f)

# ── Keyword dictionaries ───────────────────────────────────────────────────────
BODY_PARTS = {
    "brain":          ["brain","head","cranial","cranium","intracranial","cerebr","neuro","skull","orbit","sella","pituitary"],
    "spine_cervical": ["cervical","c-spine","c spine"],
    "spine_thoracic": ["thoracic spine","t-spine","t spine"],
    "spine_lumbar":   ["lumbar","l-spine","l spine","lumbosacral","sacral"],
    "chest":          ["chest","thorax","lung","pulmon","pleural","mediastin","rib","heart","coronary","cardiac","spect","nm myo","nmmyo","myocard"],
    "abdomen":        ["abdomen","abdominal","liver","hepat","pancrea","spleen","kidney","renal","adrenal","bowel","colon","rectum","gallbladder","biliary","aaa"],
    "pelvis":         ["pelvis","pelvic","bladder","prostate","uterus","ovary","abd/pel","abd pel"],
    "upper_ext":      ["shoulder","humerus","elbow","forearm","wrist","hand","finger","clavicle"],
    "lower_ext":      ["hip","femur","knee","tibia","fibula","ankle","foot","toe"],
    "breast":         ["breast","mammograph","mammo","mam "],
    "neck":           ["neck","thyroid","soft tissue neck","parotid"],
    "vascular":       ["angio","vascular","venous","arterial","carotid","doppler"],
    "bone":           ["bone density","dxa","dexa","osteo"],
}

MODALITIES = {
    "mri":        ["mri","mr ","magnetic","flair","dwi"],
    "ct":         ["ct ","cta","computed tom","cntrst","angiogram"],
    "xray":       ["xray","x-ray","radiograph","xr "," view","frontal","pa/lat"],
    "ultrasound": ["ultrasound","us ","sonograph","echo","doppler"],
    "nuclear":    ["pet","nuclear","bone scan","spect","nm ","myo perf"],
    "mammo":      ["mammo","mammograph","mam "],
}

# ── Feature helpers ────────────────────────────────────────────────────────────
def get_parts(desc: str) -> frozenset:
    d = desc.lower()
    return frozenset(p for p, kws in BODY_PARTS.items() if any(k in d for k in kws))

def get_mods(desc: str) -> frozenset:
    d = desc.lower()
    return frozenset(m for m, kws in MODALITIES.items() if any(k in d for k in kws))

def get_side(desc: str) -> str:
    d = desc.upper()
    if "BILAT" in d or "BILATERAL" in d:
        return "bilateral"
    if (" LT" in d or "LEFT" in d) and not (" RT" in d or "RIGHT" in d):
        return "left"
    if (" RT" in d or "RIGHT" in d) and not (" LT" in d or "LEFT" in d):
        return "right"
    return "unknown"

def years_apart(d1: str, d2: str) -> float:
    try:
        t1 = datetime.strptime(d1[:10], "%Y-%m-%d")
        t2 = datetime.strptime(d2[:10], "%Y-%m-%d")
        return abs((t1 - t2).days) / 365.25
    except (ValueError, TypeError):
        return 3.0

def build_features(cur_desc: str, cur_date: str, pri_desc: str, pri_date: str) -> list:
    cp = get_parts(cur_desc); pp = get_parts(pri_desc)
    cm = get_mods(cur_desc);  pm = get_mods(pri_desc)
    cs = get_side(cur_desc);  ps = get_side(pri_desc)
    yr = years_apart(cur_date, pri_date)
    po = len(cp & pp); mo = len(cm & pm)
    pu = len(cp | pp) or 1; mu = len(cm | pm) or 1
    opp  = int((cs == "left" and ps == "right") or (cs == "right" and ps == "left"))
    same = int(cs == ps and cs != "unknown")
    bi   = int(cs == "bilateral" and ps == "bilateral")
    return [
        yr / 20.0, po, po / pu, mo, mo / mu,
        opp, same, int(cur_desc == pri_desc), bi,
        int(yr <= 1), int(yr <= 3), int(yr > 10),
        int(po > 0), int(mo > 0), int(po > 0 and mo > 0),
    ]

# ── Input sanitisers ───────────────────────────────────────────────────────────
def safe_desc(study: dict) -> str:
    return str(study.get("study_description") or "").upper()

def safe_date(study: dict) -> str:
    return str(study.get("study_date") or "")

# ── Targeted clinical rules ────────────────────────────────────────────────────
def is_mammography(desc: str) -> bool:
    """True for any mammography or breast imaging study."""
    d = desc.lower()
    return any(k in d for k in ["mam ", "mammo", "mammograph", "breast"])

def is_chest_xray(desc: str) -> bool:
    """True for chest X-ray studies (excludes mammography)."""
    d = desc.lower()
    return any(k in d for k in ["chest", "cxr", "frontal", "pa/lat", "thorax"]) \
           and not is_mammography(desc)

def is_ct_chest(desc: str) -> bool:
    """True for CT of chest only (excludes combined chest/abdomen studies)."""
    d = desc.lower()
    return "ct " in d and "chest" in d and "abd" not in d and "pelv" not in d

def is_ct_abdomen(desc: str) -> bool:
    """True for CT of abdomen or pelvis (includes combined studies)."""
    d = desc.lower()
    return "ct " in d and any(k in d for k in ["abdomen", "pelvis", "abd/pel", "abd pel"])

def is_mam_bilateral(desc: str) -> bool:
    """True for bilateral mammography (screening or explicit bilateral)."""
    d = desc.lower()
    return is_mammography(desc) and any(k in d for k in ["bilat", "bilateral", " bi ", "screen", "3d"])

def is_mam_unilateral(desc: str) -> bool:
    """True for unilateral mammography (has L/R laterality, not bilateral)."""
    d = desc.lower()
    return (is_mammography(desc)
            and not any(k in d for k in ["bilat", "bilateral", " bi "])
            and any(k in d for k in [" lt", " rt", "left", "right"]))

def targeted_rule(cur_desc: str, pri_desc: str):
    """
    High-confidence clinical rules derived from public-split analysis.

    Returns True, False, or None (None = ML classifier decides).

    Rules:
      1. Mammography vs chest X-ray -> not relevant
         Evidence: 643 false, 3 true (99.5% accuracy)
         Rationale: breast parenchyma vs lung/mediastinum — different anatomy

      2. CT Chest vs CT Abdomen/Pelvis -> not relevant
         Evidence: 319 false, 5 true (98.5% accuracy)
         Rationale: non-overlapping anatomical regions in routine reads

      3. Bilateral mammography vs unilateral mammography -> relevant
         Evidence: 294 true, 29 false (91.0% accuracy)
         Rationale: screening reads compare against all prior breast imaging

    Clinical trade-off note: Rule 3 (91% accuracy) accepts ~9% false positives
    because showing an unnecessary prior is less harmful than missing a relevant
    prior in breast cancer screening workflows (recall > precision for safety).
    """
    if is_mammography(cur_desc) and is_chest_xray(pri_desc):   return False
    if is_chest_xray(cur_desc) and is_mammography(pri_desc):   return False
    if is_ct_chest(cur_desc) and is_ct_abdomen(pri_desc):      return False
    if is_ct_abdomen(cur_desc) and is_ct_chest(pri_desc):      return False
    if is_mam_bilateral(cur_desc) and is_mam_unilateral(pri_desc): return True
    if is_mam_unilateral(cur_desc) and is_mam_bilateral(pri_desc): return True
    return None
