"""Answer normalization, ported from the official VQA/TextVQA evaluation."""
import re

_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't", "hes": "he's",
    "im": "i'm", "isnt": "isn't", "its": "it's", "lets": "let's", "shes": "she's",
    "thats": "that's", "theres": "there's", "theyre": "they're", "wasnt": "wasn't",
    "werent": "weren't", "whats": "what's", "wheres": "where's", "wont": "won't",
    "wouldnt": "wouldn't", "youre": "you're", "youve": "you've",
}
_MANUAL_MAP = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_ARTICLES = {"a", "an", "the"}
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")
_PUNCT = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\",
          "_", "-", ">", "<", "@", "`", ",", "?", "!"]


def _process_punctuation(text: str) -> str:
    out = text
    for p in _PUNCT:
        if (p + " " in text or " " + p in text) or (re.search(_COMMA_STRIP, text) is not None):
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    out = _PERIOD_STRIP.sub("", out, re.UNICODE)
    return out


def _process_digit_article(text: str) -> str:
    out = []
    for w in text.lower().split():
        w = _MANUAL_MAP.get(w, w)
        if w in _ARTICLES:
            continue
        out.append(_CONTRACTIONS.get(w, w))
    return " ".join(out)


def normalize_answer(ans: str) -> str:
    ans = ans.replace("\n", " ").replace("\t", " ").strip().lower()
    ans = _process_punctuation(ans)
    ans = _process_digit_article(ans)
    return ans
