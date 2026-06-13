from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple


_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_SENTENCE_RE = re.compile(r"[^.!?]+(?:[.!?]+|$)")
_SPEAKER_RE = re.compile(
    r"(?i)(?:^|\s)(INTERVIEWER|INTERVIEWEE|TRANSLATION|TRANSLATOR|PARTICIPANT|RESPONDENT|MODERATOR|FACILITATOR|RESEARCHER|INT|P\d+)\s*:"
)
_SPEAKER_LABEL_RE = re.compile(
    r"(?i)\b(INTERVIEWER|INTERVIEWEE|TRANSLATION|TRANSLATOR|PARTICIPANT|RESPONDENT|MODERATOR|FACILITATOR|RESEARCHER|INT|P\d+)\s*:"
)
_LINE_NUMBER_BEFORE_SPEAKER_RE = re.compile(
    r"(?i)\b\d{1,4}\s+(?=(INTERVIEWER|INTERVIEWEE|TRANSLATION|TRANSLATOR|PARTICIPANT|RESPONDENT|MODERATOR|FACILITATOR|RESEARCHER|INT|P\d+)\s*:)"
)
_STANDALONE_LINE_NUMBER_RE = re.compile(r"(?m)^\s*\d{1,4}\s+")
_INLINE_LINE_NUMBER_RE = re.compile(r"(?<=[.!?])\s+\d{1,4}\s+(?=[A-Z\"'])")
_WHITESPACE_RE = re.compile(r"\s+")

_PARTICIPANT_SPEAKERS = {"interviewee", "translation", "translator", "participant", "respondent"}
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
    "you",
    "your",
    "they",
    "their",
    "this",
    "that",
}
_DEMOGRAPHIC_MARKERS = {
    "interview transcription participant",
    "demographic data",
    "age range",
    "gender",
    "education",
    "employment status",
    "housing",
    "last grade passed",
    "tertiary level qualifications",
}


def normalize_space(text: Any) -> str:
    return _WHITESPACE_RE.sub(" ", str(text or "")).strip()


def evidence_tokens(text: Any) -> set[str]:
    return {
        tok.lower()
        for tok in _WORD_RE.findall(str(text or ""))
        if tok.lower() not in _STOPWORDS and len(tok) > 2
    }


def _remove_demographic_lines(text: str) -> Tuple[str, List[str]]:
    flags: List[str] = []
    kept: List[str] = []
    for line in str(text or "").splitlines():
        low = normalize_space(line).lower()
        if low and any(marker in low for marker in _DEMOGRAPHIC_MARKERS):
            flags.append("removed_demographic_header")
            continue
        kept.append(line)
    cleaned = "\n".join(kept)
    cleaned2 = re.sub(
        r"(?i)\bINTERVIEW TRANSCRIPTION PARTICIPANT\s+\S+\b.*?\bDEMOGRAPHIC DATA\b",
        " ",
        cleaned,
    )
    if cleaned2 != cleaned:
        flags.append("removed_demographic_header")
    return cleaned2, flags


def clean_transcript_artifacts(text: Any) -> Tuple[str, List[str]]:
    raw = str(text or "")
    flags: List[str] = []
    cleaned, demo_flags = _remove_demographic_lines(raw)
    flags.extend(demo_flags)
    if _SPEAKER_LABEL_RE.search(cleaned):
        flags.append("removed_speaker_labels")
    cleaned = _LINE_NUMBER_BEFORE_SPEAKER_RE.sub("", cleaned)
    cleaned = _STANDALONE_LINE_NUMBER_RE.sub("", cleaned)
    cleaned = _SPEAKER_LABEL_RE.sub(" ", cleaned)
    cleaned2 = _INLINE_LINE_NUMBER_RE.sub(" ", cleaned)
    if cleaned2 != cleaned or re.search(r"(?m)^\s*\d{1,4}\s+", raw):
        flags.append("removed_line_numbers")
    cleaned = normalize_space(cleaned2)
    return cleaned, sorted(set(flags))


def _speaker_blocks(text: Any) -> List[Tuple[str, str]]:
    raw = str(text or "")
    matches = list(_SPEAKER_RE.finditer(raw))
    blocks: List[Tuple[str, str]] = []
    for idx, match in enumerate(matches):
        speaker = (match.group(1) or "").lower()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw)
        block = raw[start:end].strip()
        if block:
            blocks.append((speaker, block))
    return blocks


def _is_participant_speaker(speaker: str) -> bool:
    low = speaker.lower()
    return low in _PARTICIPANT_SPEAKERS or bool(re.fullmatch(r"p\d+", low))


def _sentence_windows(text: str, *, max_sentences: int = 3) -> Iterable[str]:
    sentences = [normalize_space(match.group(0)) for match in _SENTENCE_RE.finditer(text) if normalize_space(match.group(0))]
    if not sentences:
        cleaned = normalize_space(text)
        if cleaned:
            yield cleaned
        return
    for idx in range(len(sentences)):
        for size in range(1, max_sentences + 1):
            window = " ".join(sentences[idx : idx + size]).strip()
            if window:
                yield window


def _clip_to_sentence_window(text: str, *, max_chars: int, max_sentences: int = 3) -> str:
    cleaned = normalize_space(text)
    sentences = [normalize_space(match.group(0)) for match in _SENTENCE_RE.finditer(cleaned) if normalize_space(match.group(0))]
    if sentences:
        kept: List[str] = []
        for sentence in sentences[:max_sentences]:
            candidate = " ".join(kept + [sentence]).strip()
            if len(candidate) > max_chars and kept:
                break
            if len(candidate) > max_chars:
                return candidate[:max_chars].rsplit(" ", 1)[0].rstrip()
            kept.append(sentence)
        if kept:
            return " ".join(kept).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rsplit(" ", 1)[0].rstrip()


def _score_quote_window(window: str, query_tokens: set[str]) -> float:
    tokens = evidence_tokens(window)
    if not tokens:
        return 0.0
    overlap = len(tokens & query_tokens) if query_tokens else 0
    # Prefer substantive participant speech over very short yes/no fragments.
    length_bonus = min(len(tokens), 45) / 45.0
    question_penalty = 0.25 if "?" in window else 0.0
    return overlap * 2.0 + length_bonus - question_penalty


def _candidate_windows_from_text(text: Any) -> List[str]:
    blocks = _speaker_blocks(text)
    raw_candidates: List[str] = []
    if blocks:
        for speaker, block in blocks:
            if _is_participant_speaker(speaker):
                raw_candidates.append(block)
    if not raw_candidates:
        raw_candidates.append(str(text or ""))

    candidates: List[str] = []
    for raw in raw_candidates:
        cleaned, _flags = clean_transcript_artifacts(raw)
        if not cleaned:
            continue
        candidates.extend(_sentence_windows(cleaned, max_sentences=3))
    return candidates


def _best_clean_window(text: Any, query_text: Any, *, max_chars: int) -> Tuple[str, List[str]]:
    query_tokens = evidence_tokens(query_text)
    prepared: List[Tuple[str, List[str]]] = []
    for window in _candidate_windows_from_text(text):
        cleaned, clean_flags = clean_transcript_artifacts(window)
        cleaned = _clip_to_sentence_window(cleaned, max_chars=max_chars)
        if len(evidence_tokens(cleaned)) >= 4:
            prepared.append((cleaned, clean_flags))
    has_non_question = any("?" not in cleaned for cleaned, _flags in prepared)
    best = ""
    best_score = -1.0
    flags: List[str] = []
    for cleaned, clean_flags in prepared:
        if has_non_question and "?" in cleaned:
            continue
        score = _score_quote_window(cleaned, query_tokens)
        if score > best_score:
            best = cleaned
            best_score = score
            flags = clean_flags
    if best:
        return best, flags
    cleaned, clean_flags = clean_transcript_artifacts(text)
    return _clip_to_sentence_window(cleaned, max_chars=max_chars), clean_flags


def build_clean_quote_evidence(
    *,
    transcript: Any,
    segment_number: Any,
    raw_text: Any,
    query_text: Any,
    provided_quote: Any = "",
    source: str,
    max_quote_chars: int = 650,
    max_context_chars: int = 900,
) -> Optional[Dict[str, Any]]:
    """Return a clean quote evidence object, or None when no substantive quote can be built."""
    flags: List[str] = []
    quote = ""
    if str(provided_quote or "").strip():
        quote, flags = _best_clean_window(provided_quote, query_text, max_chars=max_quote_chars)
        if len(evidence_tokens(quote)) < 4:
            quote = ""
    if not quote:
        quote, fallback_flags = _best_clean_window(raw_text, query_text, max_chars=max_quote_chars)
        flags.extend(fallback_flags)
        flags.append("selected_from_source_context")
    quote = normalize_space(quote)
    if not quote or len(evidence_tokens(quote)) < 4:
        return None

    source_context, context_flags = clean_transcript_artifacts(raw_text)
    flags.extend(context_flags)
    source_context = _clip_to_sentence_window(source_context, max_chars=max_context_chars, max_sentences=5)
    try:
        seg_num = int(segment_number or 0)
    except Exception:
        seg_num = 0
    return {
        "transcript": str(transcript or ""),
        "segment_number": seg_num,
        "segment_id": f"{transcript}#{seg_num}" if transcript else f"#{seg_num}",
        "quote": quote,
        "source_context": source_context,
        "source": source,
        "cleanup_flags": sorted(set(flags)),
    }


def quote_exact_or_clean_match(quote: Any, raw_text: Any) -> bool:
    q = normalize_space(quote)
    if not q:
        return False
    source = normalize_space(raw_text)
    if q and q in source:
        return True
    clean_source, _flags = clean_transcript_artifacts(raw_text)
    return q in clean_source
