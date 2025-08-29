# app/utils/text.py
import re

# Bản mở rộng viết tắt → câu đầy đủ đơn giản (bạn có thể bổ sung thêm pattern)
ABBR_MAP = [
    (r"\bmk\b|\bmik\b|\bmh\b", "mình"),
    (r"\b(k|kh|ko)\b", "không"),
    (r"\bthik\b|\bthich\b", "thích"),
    (r"\btrc\b|\btrk\b", "trước"),
    (r"\bdc\b|\bdk\b|\bduoc\b", "được"),
    (r"\bng\b", "người"),
    (r"\bcg\b|\bcug\b", "cũng"),
    (r"\bvs\b", "với"),
    (r"\bqa?\b", "quá"),
    (r"<3", "yêu"),
    (r"\^\^", "hihi"),
]

def expand_text(s: str) -> str:
    t = str(s).strip().lower()
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)  # co giãn ký tự kéo dài
    for pat, rep in ABBR_MAP:
        t = re.sub(pat, rep, t)
    t = re.sub(r"\s+", " ", t).strip()
    return t
