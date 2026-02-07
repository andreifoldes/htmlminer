"""
Formatting helpers for output normalization and prompt instructions.
"""

from __future__ import annotations

import json
import re

DEFAULT_BULLET = "-"
LIST_OUTPUT_FORMATS = {"list", "bullet_list"}

_BULLET_MARKERS = (
    "\u2022",  # bullet
    "-",
    "*",
    "\u2013",  # en dash
    "\u2014",  # em dash
    "\u00b7",  # middle dot
)


def is_list_output_format(output_format: str) -> bool:
    if not isinstance(output_format, str):
        return False
    return output_format.strip().lower() in LIST_OUTPUT_FORMATS


def list_output_instruction(bullet: str = DEFAULT_BULLET) -> str:
    bullet = (bullet or DEFAULT_BULLET).strip() or DEFAULT_BULLET
    return (
        f"Output format: bullet list using '{bullet}' prefix. "
        "One item per line. "
        "No lead-in sentence or inline separators."
    )


def normalize_bullet_list(text: str, bullet: str = DEFAULT_BULLET) -> str:
    if not isinstance(text, str):
        return text
    raw = text.strip()
    if not raw:
        return raw

    bullet = (bullet or DEFAULT_BULLET).strip() or DEFAULT_BULLET

    items = _extract_list_items(raw)
    if not items:
        return raw

    return "\n".join(f"{bullet} {item}" for item in items)


def _extract_list_items(raw: str) -> list[str]:
    # JSON array support (common when models return lists)
    if raw.startswith("[") and raw.endswith("]"):
        try:
            data = json.loads(raw)
        except Exception:
            data = None
        if isinstance(data, list):
            items = [str(item).strip() for item in data if str(item).strip()]
            if items:
                return items

    # Inline bullet markers (e.g., "• item1 • item2")
    if "\u2022" in raw or "\u00b7" in raw:
        parts = re.split(r"[\u2022\u00b7]", raw)
        items = [_strip_bullet_lead(p) for p in parts if p.strip()]
        if len(items) > 1:
            return items

    # Line-based bullets
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if lines and any(_line_starts_with_bullet(line) for line in lines):
        items = [_strip_bullet_lead(line) for line in lines]
        return [item for item in items if item]

    # Inline separators without newlines
    if "\n" not in raw:
        if ";" in raw:
            parts = [p.strip() for p in raw.split(";") if p.strip()]
            if len(parts) > 1:
                parts[0] = _strip_leadin_colon(parts[0])
                return [p for p in parts if p]
        if " - " in raw:
            parts = [p.strip() for p in raw.split(" - ") if p.strip()]
            if len(parts) > 1:
                parts[0] = _strip_leadin_colon(parts[0])
                items = [_strip_bullet_lead(p) for p in parts]
                return [item for item in items if item]

    return []


def _line_starts_with_bullet(line: str) -> bool:
    stripped = line.lstrip()
    if not stripped:
        return False
    if stripped[0] in _BULLET_MARKERS:
        return True
    return bool(re.match(r"^\d+[\.\)]\s+", stripped))


def _strip_bullet_lead(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""

    for marker in _BULLET_MARKERS:
        if cleaned.startswith(marker):
            cleaned = cleaned[len(marker):].lstrip()
            break

    cleaned = re.sub(r"^\d+[\.\)]\s+", "", cleaned).strip()
    return cleaned


def _strip_leadin_colon(text: str) -> str:
    if ":" not in text:
        return text
    lead, tail = text.rsplit(":", 1)
    tail = tail.strip()
    return tail if tail else text
