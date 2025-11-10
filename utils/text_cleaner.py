import orjson
from typing import List
from google.genai.client import Models


def emoji_to_codepoints(text: str) -> str:
            """Convert each character to U+XXXX if it's non-ASCII (e.g., emojis)."""
            result = []
            for ch in text:
                if ord(ch) > 127:  # non-ASCII
                    result.append(f"U+{ord(ch):04X}")  # Unicode codepoint
                else:
                    result.append(ch)
            return "".join(result)

def codepoints_to_emoji(unicode_str: str) -> str:
    # Split the string by "U+"
    parts = unicode_str.split("U+")
    # Convert each hex code to emoji if valid
    emojis = ""
    for p in parts:
        if p.strip() == "":
            continue
        try:
            emojis += chr(int(p, 16))
        except ValueError:
            emojis += p  # fallback if not valid hex
    return emojis

def truncate_by_tokens(texts: List[dict], model: Models, max_tokens: int):
    lo, hi = 0, len(texts)
    best_fit = 0

    while lo <= hi:
        mid = (lo + hi) // 2
        temp_json = orjson.dumps(texts[:mid])
        token_info = model.count_tokens(model=model, contents=temp_json)

        if token_info.total_tokens <= max_tokens:
            best_fit = mid  # valid, but maybe we can take more
            lo = mid + 1
        else:
            hi = mid - 1

    return texts[:best_fit], temp_json, token_info.total_tokens