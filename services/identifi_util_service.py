from langdetect import detect
import emoji
import re
from urllib.parse import urlparse



class IdentifiScoreUtil:
    """Helpers for identifi scoring"""

    SPAM_DOMAINS = {
    "grabify.link", "iplogger.org", "adf.ly", "shorte.st", "tinyurl.com",
    }

    SPAM_TLDS = {".top", ".xyz", ".click", ".buzz", ".loan", ".work", ".online"}

    SPAM_KEYWORDS = {
        "free", "promo", "bonus", "airdrop", "giveaway", "click", "claim",
        "win", "reward", "discount"
    }

    @staticmethod
    def normalize(value, min_val, max_val):
        return (value - min_val) / (max_val - min_val + 1e-9)

    @staticmethod
    def detect_language_safe(text):
        try:
            return detect(text)
        except:
            return "en"

    @staticmethod
    def tokenize(text):
        """Split text into words, hashtags, and punctuation."""
        return re.findall(r"\w+|#\w+|[^\w\s]", text, re.UNICODE)

    @staticmethod
    def count_sentences(text):
        """Count sentence boundaries safely."""
        return max(1, len(re.findall(r"[.!?]+", text)))

    @staticmethod
    def emoji_count(text):
        return sum(1 for ch in text if ch in emoji.EMOJI_DATA)

    @staticmethod
    def vowel_ratio(word):
        vowels = "aeiou"
        if not word:
            return 1.0
        v = sum(1 for c in word.lower() if c in vowels)
        return v / len(word)

    @staticmethod
    def extract_urls(text: str):
        url_regex = r'(https?://\S+)'
        return re.findall(url_regex, text)

    @staticmethod
    def is_spam_url(url: str):
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Check standard spam domains
        if domain in IdentifiScoreUtil.SPAM_DOMAINS:
            return True, 0.8  # heavy spam

        # Check TLD
        for tld in IdentifiScoreUtil.SPAM_TLDS:
            if domain.endswith(tld):
                return True, 0.5

        # Check spam keywords in the full URL
        for kw in IdentifiScoreUtil.SPAM_KEYWORDS:
            if kw in url.lower():
                return True, 0.4

        # Looks normal
        return False, 0.0

    @staticmethod
    def clean_tweet(text: str):
        text = text.lower()
        text = re.sub(r"\brt\b", "", text)
        text = re.sub(r"@[A-Za-z0-9_]+", "", text) # remove mentions @username
        text = re.sub(r"http\S+|www\.\S+", "", text)
        text = re.sub(r"[^a-z0-9\s]", " ", text) # remove emojis & non-alphanumeric except whitespace
        text = re.sub(r"\s+", " ", text).strip() # collapse extra spaces
        return text