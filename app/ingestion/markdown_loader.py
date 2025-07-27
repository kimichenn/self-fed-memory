"""Load a Markdown file → List[dict] with accurate timestamps.

Priority rules
--------------
1. `created:` front-matter (e.g. “Jun 11, 2024 at 9:40 AM”)
2. Date encoded in the **filename**  (yyyy-mm-dd.*) - time fixed to 12:00
3. File-system mtime.
If front-matter *and* filename disagree on the **date**, we keep
the filename date (12 PM) to avoid silent conflicts.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Any
import uuid

import frontmatter

from app.utils.text_splitter import split_markdown

__all__ = ["parse_markdown_file"]

# --------------------------------------------------------------------- #
# Helpers                                                               #
# --------------------------------------------------------------------- #
_FM_CREATED_RE = re.compile(
    r"^created:\s*([A-Za-z]{3}\s+\d{1,2},\s+\d{4})"
    r"(?:\s+at\s+(\d{1,2}:\d{2}\s*[AP]M))?",
    re.IGNORECASE | re.MULTILINE,
)

_FILENAME_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")


def _frontmatter_datetime(text: str) -> datetime | None:
    """Parse the `created:` line; returns tz-naive UTC."""
    m = _FM_CREATED_RE.search(text)
    if not m:
        return None

    date_part, time_part = m.groups()
    fmt = "%b %d, %Y" + (" %I:%M %p" if time_part else "")
    try:
        return datetime.strptime(f"{date_part} {time_part or ''}".strip(), fmt)
    except ValueError:
        return None


def _filename_datetime(path: Path) -> datetime | None:
    """Extract yyyy-mm-dd from the filename; attach 12:00."""
    m = _FILENAME_DATE_RE.match(path.stem)
    if not m:
        return None
    try:
        return datetime.fromisoformat(f"{m.group(1)}T12:00:00")
    except ValueError:
        return None


# --------------------------------------------------------------------- #
# Public API                                                            #
# --------------------------------------------------------------------- #
def parse_markdown_file(path: str | Path) -> list[dict[str, Any]]:
    """Return *chunked* representation of one Markdown file."""
    path = Path(path)
    post = frontmatter.load(path)
    content = post.content
    fm_created = post.metadata.get("created")

    fm_dt = None
    if fm_created:
        if isinstance(fm_created, datetime):
            fm_dt = fm_created
        elif isinstance(fm_created, str):
            try:
                fm_dt = datetime.fromisoformat(fm_created)
            except ValueError:
                try:
                    # Attempt to parse "Month Day, Year at HH:MM AM/PM"
                    fm_dt = datetime.strptime(fm_created, "%b %d, %Y at %I:%M %p")
                except ValueError:
                    # Fallback for "Month Day, Year"
                    try:
                        fm_dt = datetime.strptime(fm_created, "%b %d, %Y")
                    except ValueError:
                        fm_dt = None
        else:
            fm_dt = None

    fn_dt = _filename_datetime(path)
    mtime_dt = datetime.fromtimestamp(path.stat().st_mtime)

    # Resolve conflicts
    if fn_dt and fm_dt and fn_dt.date() != fm_dt.date():
        chosen_ts = fn_dt  # filename wins on date, 12 PM time
    else:
        chosen_ts = fm_dt or fn_dt or mtime_dt

    chunks = split_markdown(content)
    return [
        {
            "id": str(uuid.uuid4()),
            "content": chunk,
            "created_at": chosen_ts.isoformat(),
            "source": str(path),
        }
        for chunk in chunks
    ]
