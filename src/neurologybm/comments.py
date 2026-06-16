"""Private comment extraction helpers for case challenge discussion pages."""

from __future__ import annotations

import html
import re
from pathlib import Path

from .deepseek import assert_private_path


TAG_RE = re.compile(r"<[^>]+>")
SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
SPACE_RE = re.compile(r"[ \t\r\f\v]+")
BLANK_RE = re.compile(r"\n{3,}")


def extract_visible_text_from_html(html_text: str) -> str:
    cleaned = SCRIPT_STYLE_RE.sub("", html_text)
    cleaned = re.sub(r"</(p|div|li|tr|h[1-6]|section|article|blockquote)>", "\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<br\s*/?>", "\n", cleaned, flags=re.IGNORECASE)
    cleaned = TAG_RE.sub(" ", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = SPACE_RE.sub(" ", cleaned)
    cleaned = "\n".join(line.strip() for line in cleaned.splitlines())
    return BLANK_RE.sub("\n\n", cleaned).strip()


def extract_discussion_texts(raw_html_dir: Path, out_dir: Path) -> list[dict[str, str]]:
    assert_private_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for html_path in sorted(raw_html_dir.glob("*.html")):
        text = extract_visible_text_from_html(html_path.read_text(encoding="utf-8", errors="replace"))
        row = {
            "source_html": str(html_path),
            "discussion_id": html_path.stem,
            "text_path": str(out_dir / f"{html_path.stem}.txt"),
            "char_count": str(len(text)),
        }
        Path(row["text_path"]).write_text(text + "\n", encoding="utf-8")
        rows.append(row)
    return rows

