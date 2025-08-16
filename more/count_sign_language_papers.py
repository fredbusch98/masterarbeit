"""
Queries the Semantic Scholar Graph API to report the TOTAL
(raw) hit count for each of:
  • Sign Language Production / Generation
  • Sign Language Recognition
  • Sign Language Translation
"""
import requests
import time

# ─────────── USER CONFIG ───────────

KEYWORDS = {
    "Sign Language Production": [
        '"sign language production"',
        '"sign-language-production"',
        '"sign-language production"',
    ],
    "Sign Language Generation": [
        '"sign language generation"',
        '"sign-language-generation"',
        '"sign-language generation"',
    ],
    "Sign Language Recognition": [
        '"sign language recognition"',
        '"sign-language-recognition"',
        '"sign-language recognition"',
    ],
    "Sign Language Translation": [
        '"sign language translation"',
        '"sign-language-translation"',
        '"sign-language translation"',
    ],
}

API_URL       = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS        = "paperId"   # required by the API, but we only care about `total`
MAX_BACKOFF   = 64          # cap for exponential back-off (seconds)


# ─────────── HELPERS ───────────

def request_with_backoff(url, params):
    backoff = 1.0
    while True:
        resp = requests.get(url, params=params)
        if resp.status_code == 429:
            print(f"   ⚠️ Rate limited—backing off for {backoff:.1f}s …")
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)
            continue
        resp.raise_for_status()
        return resp.json()


# ─────────── CORE LOGIC ───────────

def get_raw_hit_count(phrases, label):
    print(f"\n🔍 Querying: **{label}**")
    data = request_with_backoff(API_URL, {
        "query":  " OR ".join(phrases),
        "fields": FIELDS,
        "limit":  1,
        "cursor": "*",
    })
    total = data.get("total", 0)
    print(f"🎯 Total raw hits for **{label}**: {total} 📚")
    return total


def main():
    print("🎉 Starting raw hit‐count report\n")
    for label, phrases in KEYWORDS.items():
        get_raw_hit_count(phrases, label)
    print("\n✅ All done!")

if __name__ == "__main__":
    main()
