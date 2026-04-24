import os
import json
import hashlib
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from groq import Groq
import uvicorn

app = FastAPI()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

cache = {}

def make_cache_key(current_study: dict, prior_study: dict) -> str:
    key = f"{current_study.get('study_description','')}|{prior_study.get('study_description','')}|{prior_study.get('study_date','')}"
    return hashlib.md5(key.encode()).hexdigest()

def predict_relevance_batch(current_study: dict, prior_studies: list) -> list:
    results = []
    uncached = []
    uncached_indices = []

    for i, prior in enumerate(prior_studies):
        ck = make_cache_key(current_study, prior)
        if ck in cache:
            results.append((i, cache[ck]))
        else:
            uncached.append(prior)
            uncached_indices.append(i)
            results.append((i, None))

    if uncached:
        priors_text = "\n".join([
            f"{j+1}. study_id={p['study_id']} | desc={p.get('study_description','N/A')} | date={p.get('study_date','N/A')}"
            for j, p in enumerate(uncached)
        ])

        prompt = f"""You are a radiology AI assistant. A radiologist is reading a current exam and needs to know which prior exams are relevant to review.

Current exam:
- Description: {current_study.get('study_description', 'N/A')}
- Date: {current_study.get('study_date', 'N/A')}

Prior exams to evaluate:
{priors_text}

For each prior exam, determine if it is relevant for the radiologist to review while reading the current exam.

A prior exam is relevant if:
- It is the same or similar body part/region
- It is the same or related modality (MRI/CT/X-ray) that provides useful comparison
- It shows the same or related pathology/condition
- It is recent enough to be clinically meaningful

Return ONLY a JSON array with true/false for each prior in the same order, like:
[true, false, true]

No explanation, just the JSON array."""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=512,
        )

        raw = response.choices[0].message.content.strip()
        try:
            raw = raw[raw.index('['):raw.rindex(']')+1]
            predictions = json.loads(raw)
        except Exception:
            predictions = [True] * len(uncached)

        for idx, (orig_i, prior) in enumerate(zip(uncached_indices, uncached)):
            val = bool(predictions[idx]) if idx < len(predictions) else True
            ck = make_cache_key(current_study, prior)
            cache[ck] = val
            results[orig_i] = (orig_i, val)

    return [r[1] for r in sorted(results, key=lambda x: x[0])]


@app.post("/predict")
async def predict(request: Request):
    body = await request.json()
    cases = body.get("cases", [])
    predictions = []

    for case in cases:
        case_id = case["case_id"]
        current_study = case["current_study"]
        prior_studies = case.get("prior_studies", [])

        relevances = predict_relevance_batch(current_study, prior_studies)

        for prior, is_relevant in zip(prior_studies, relevances):
            predictions.append({
                "case_id": case_id,
                "study_id": prior["study_id"],
                "predicted_is_relevant": is_relevant
            })

    return JSONResponse(content={"predictions": predictions})


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
