"""Multilingual CJK + Latin tokenizer, entity-slot parser, and deterministic embeddings."""

from __future__ import annotations

import hashlib
import re
import numpy as np


_LATIN_TOKEN_RE = re.compile(r"[a-z0-9_+#.-]{2,}")
_CJK_SPAN_RE = re.compile(r"[\u4e00-\u9fff]+")
_ENTITY_PROPER_RE = re.compile(
    r"\b(?:[A-Z][a-zA-Z0-9_-]{1,}(?:\s+[A-Z][a-zA-Z0-9_-]{1,})*|[A-Z]{2,}[0-9-]*)\b"
)
_QUOTED_ENTITY_RE = re.compile(r"[「『《“\"']([^」』》”\"']{2,24})[」』》”\"']")

_EN_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "that", "this", "from", "have", "has", "had",
        "are", "was", "were", "will", "would", "could", "should", "what", "when",
        "where", "which", "who", "whom", "whose", "why", "how", "into", "onto",
        "about", "please", "can", "you", "your", "our", "their", "his", "her",
        "its", "they", "them", "been", "being", "also", "just", "only", "very",
    }
)

_ZH_GRAMMAR_PARTICLES = frozenset(
    {"的", "了", "吗", "呢", "吧", "啊", "着", "请问", "帮我", "一下", "什么", "怎么", "哪些"}
)

_FIRST_PERSON_MARKERS = ("我", "本人", "我的", "咱们", "用户")

# Bilingual synonym canonicalization table bridging cross-lingual and paraphrase gaps
# in offline deterministic embedding & lexical indexing mode.
_CONCEPT_ALIASES: dict[str, str] = {
    "住址": "city",
    "居住": "city",
    "住在": "city",
    "搬到": "city",
    "定居": "city",
    "城市": "city",
    "lives": "city",
    "living": "city",
    "moved": "city",
    "relocated": "city",
    "resides": "city",
    "location": "city",
    "预算": "budget",
    "经费": "budget",
    "报销上限": "budget",
    "开销": "budget",
    "budget": "budget",
    "allowance": "budget",
    "会议": "meeting",
    "开会": "meeting",
    "例会": "meeting",
    "组会": "meeting",
    "日程": "meeting",
    "meeting": "meeting",
    "standup": "meeting",
    "sync": "meeting",
    "偏好": "preference",
    "喜欢": "preference",
    "习惯": "preference",
    "prefer": "preference",
    "prefers": "preference",
    "favorite": "preference",
    "截止": "deadline",
    "死线": "deadline",
    "交稿": "deadline",
    "deadline": "deadline",
    "due": "deadline",
    "负责人": "owner",
    "对接人": "owner",
    "审批人": "owner",
    "owner": "owner",
    "approver": "owner",
    "发票": "invoice",
    "报销": "reimbursement",
    "reimburse": "reimbursement",
    "invoice": "invoice",
}

# Patterns for automatic (subject, single-valued slot, slot_value) extraction
_SLOT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "city",
        re.compile(
            r"(?P<subj>用户|我|本人|[A-Z][a-zA-Z]+|[\u4e00-\u9fff]{2,4})"
            r"(?:目前|现在|已经|已)?(?:住在|居住在|定居在|搬到了|迁居至|常驻|lives\s+in|moved\s+to|relocated\s+to)\s*"
            r"(?P<val>[A-Za-z]+(?:\s+[A-Za-z]+)?|[\u4e00-\u9fff]{2,10})",
            re.IGNORECASE,
        ),
    ),
    (
        "meeting_time",
        re.compile(
            r"(?P<subj>[A-Za-z0-9_-]+(?:\s+sync|\s+standup|\s+meeting)?|[\u4e00-\u9fffA-Za-z0-9_-]{2,14}(?:例会|周会|组会|评审会|会议))"
            r"(?:时间)?(?:改到|调整为|定在|安排在|推迟到|moved\s+to|scheduled\s+for|rescheduled\s+to)\s*"
            r"(?P<val>(?:周[一二三四五六日天]|星期[一二三四五六日天]|Mon|Tue|Wed|Thu|Fri)[A-Za-z\u4e00-\u9fff0-9:：\s-]{1,18}|\d{1,2}[:：]\d{2})",
            re.IGNORECASE,
        ),
    ),
    (
        "budget",
        re.compile(
            r"(?P<subj>[A-Za-z0-9_-]+项目|[\u4e00-\u9fffA-Za-z0-9_-]{2,14}(?:项目|差旅|算力|采购))?"
            r"(?:预算|经费|上限|budget)\s*(?:调整为|改为|设定为|为|是|is|set\s+to|updated\s+to)\s*"
            r"(?P<val>(?:RMB|USD|CNY|\$|￥|¥)?\s*\d[\d,.]*\s*(?:万|元|k|K|USD|CNY)?)",
            re.IGNORECASE,
        ),
    ),
    (
        "owner",
        re.compile(
            r"(?P<subj>[\u4e00-\u9fffA-Za-z0-9_-]{2,16}(?:模块|发票|审批|项目|服务))"
            r"(?:的)?(?:负责人|对接人|审批人|owner|approver)(?:换成了|变更为|改为|是|is)\s*"
            r"(?P<val>[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?|[\u4e00-\u9fff]{2,6})",
            re.IGNORECASE,
        ),
    ),
]


def _latin_stem(word: str) -> str:
    """Lightweight deterministic English suffix stripper."""
    if len(word) <= 3:
        return word
    for suffix in ("ingly", "edly", "ments", "ment", "ings", "ing", "ed", "ies", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            if suffix == "ies":
                return word[:-3] + "y"
            return word[: -len(suffix)]
    return word


def multilingual_tokens(text: str) -> list[str]:
    """Extract normalized Latin stems, CJK bigrams/trigrams, and canonical concept tags."""
    if not text:
        return []
    lowered = text.lower()
    tokens: list[str] = []

    if re.search(r"\b(i|my|me|myself)\b", lowered) or any(m in text for m in _FIRST_PERSON_MARKERS):
        tokens.append("subj_self")

    for raw in _LATIN_TOKEN_RE.findall(lowered):
        if raw in _EN_STOPWORDS:
            continue
        stem = _latin_stem(raw)
        tokens.append(stem)
        if raw in _CONCEPT_ALIASES:
            tokens.append(f"concept_{_CONCEPT_ALIASES[raw]}")
        elif stem in _CONCEPT_ALIASES:
            tokens.append(f"concept_{_CONCEPT_ALIASES[stem]}")

    for span in _CJK_SPAN_RE.findall(text):
        for phrase, canonical in _CONCEPT_ALIASES.items():
            if len(phrase) >= 2 and phrase in span:
                tokens.append(f"concept_{canonical}")

        # Strip only pure grammatical particles at boundaries while preserving content characters (e.g. 上海, 北京)
        cleaned = "".join(ch for ch in span if ch not in {"的", "了", "吗", "呢", "吧", "啊", "着"})
        if not cleaned:
            continue
        if len(cleaned) == 1:
            if cleaned not in _ZH_GRAMMAR_PARTICLES:
                tokens.append(cleaned)
            continue
        for idx in range(len(cleaned) - 1):
            bg = cleaned[idx : idx + 2]
            if bg not in _ZH_GRAMMAR_PARTICLES:
                tokens.append(bg)
                if bg in _CONCEPT_ALIASES:
                    tokens.append(f"concept_{_CONCEPT_ALIASES[bg]}")

    return tokens


def fts5_segmented_text(text: str) -> str:
    """Convert multilingual text into whitespace-delimited tokens suitable for SQLite FTS5."""
    toks = multilingual_tokens(text)
    safe = [re.sub(r"[^a-z0-9_\u4e00-\u9fff]", "", t) for t in toks]
    return " ".join(t for t in safe if t)


def build_fts5_match_query(query: str, max_terms: int = 16) -> str:
    """Build a safe OR-joined SQLite FTS5 MATCH expression from user query text."""
    seen: list[str] = []
    for tok in multilingual_tokens(query):
        cleaned = re.sub(r"[^a-z0-9_\u4e00-\u9fff]", "", tok)
        if len(cleaned) < 2 and not ("\u4e00" <= cleaned <= "\u9fff"):
            continue
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
        if len(seen) >= max_terms:
            break
    if not seen:
        return ""
    return " OR ".join(f'"{term}"' for term in seen)


class DeterministicHashEmbedder:
    """Signed multi-hash feature projection producing L2-normalized dense embeddings.

    Used automatically when no live embedding API key is configured, providing
    O(|tokens|) deterministic vector representations across English and Chinese.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim < 32:
            raise ValueError("Embedding dimension must be >= 32")
        self.dim = dim

    def embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        toks = multilingual_tokens(text)
        if not toks:
            vec[0] = 1.0
            return vec

        # Unigram + token-bigram features with sublinear TF weighting
        features: dict[str, float] = {}
        for tok in toks:
            if tok == "subj_self":
                weight = 2.4
            elif tok.startswith("concept_"):
                weight = 1.8
            else:
                weight = 1.0
            features[tok] = features.get(tok, 0.0) + weight
        for idx in range(len(toks) - 1):
            pair = f"{toks[idx]}::{toks[idx + 1]}"
            features[pair] = features.get(pair, 0.0) + 0.6

        for feat, count in features.items():
            tf_weight = 1.0 + float(np.log1p(count))
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=16).digest()
            # Project each feature into 2 signed coordinates to reduce hash collision variance
            h1 = int.from_bytes(digest[0:4], "little") % self.dim
            s1 = 1.0 if (digest[4] & 1) == 0 else -1.0
            h2 = int.from_bytes(digest[8:12], "little") % self.dim
            s2 = 1.0 if (digest[12] & 1) == 0 else -1.0
            vec[h1] += s1 * tf_weight
            vec[h2] += s2 * (0.5 * tf_weight)

        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec /= norm
        else:
            vec[0] = 1.0
        return vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self.embed_one(t) for t in texts])


def extract_entities(text: str) -> list[str]:
    """Extract canonical named entities, ego-node (`user`), project codes, and identifiers."""
    entities: list[str] = []
    lowered = text.lower()
    if re.search(r"\b(i|my|me|myself)\b", lowered) or any(m in text for m in _FIRST_PERSON_MARKERS):
        entities.append("user")

    for match in _QUOTED_ENTITY_RE.findall(text):
        cleaned = match.strip()
        if cleaned and cleaned not in entities:
            entities.append(cleaned)

    for match in _ENTITY_PROPER_RE.findall(text):
        cleaned = match.strip()
        if cleaned.lower() not in _EN_STOPWORDS and cleaned not in entities:
            entities.append(cleaned)

    # Capture project/code patterns such as Project-Alpha, PRJ-204, Qwen3-8B
    for code in re.findall(r"\b[A-Za-z]{2,10}[-_]\d{1,6}[A-Za-z0-9_-]*\b", text):
        if code not in entities:
            entities.append(code)

    # Capture common Chinese role/project suffix entities (e.g., 星辰项目, 差旅报销模块)
    for zh_ent in re.findall(
        r"([\u4e00-\u9fff]{2,6}(?:项目|系统|模块|团队|实验室|周会|例会|评审会))", text
    ):
        for prefix in ("关于", "查询", "请问", "目前", "现在", "最新", "我们", "更新"):
            if zh_ent.startswith(prefix) and len(zh_ent) - len(prefix) >= 3:
                zh_ent = zh_ent[len(prefix) :]
        if zh_ent not in entities:
            entities.append(zh_ent)

    return entities[:16]


def infer_subject_slot_value(text: str) -> tuple[str | None, str | None, str | None]:
    """Infer `(subject, slot, slot_value)` for single-valued temporal state facts."""
    for slot_name, pattern in _SLOT_PATTERNS:
        m = pattern.search(text)
        if m:
            subj = (m.group("subj") or "user").strip()
            if subj in {"我", "本人", "用户", "i", "me", "user"}:
                subj = "user"
            val = m.group("val").strip(" ，。,.!！;；")
            return subj.lower(), slot_name, val
    return None, None, None
