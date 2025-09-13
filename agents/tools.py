from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def write_json_txt(base_dir: Path | str, filename: str, payload: Any) -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"
    out = base / filename
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out


def build_segment_maps(segments: List[Dict[str, Any]]) -> Tuple[Dict[Tuple[str, int], Dict[str, str]], Dict[str, Dict[str, str]]]:
    """Return: (by_key map, per_transcript map). by_key key: (transcript, segment_number) -> {raw, ivw}
    per_transcript: transcript -> {segment_number_str -> raw}
    """
    by_key: Dict[Tuple[str, int], Dict[str, str]] = {}
    per_tx: Dict[str, Dict[str, str]] = {}
    for s in segments:
        t = s.get("transcript")
        n = int(s.get("segment_number"))
        raw = s.get("text", "")
        by_key[(t, n)] = {"raw": raw, "ivw": raw}
        per_tx.setdefault(t, {})[str(n)] = raw
    return by_key, per_tx

