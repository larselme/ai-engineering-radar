from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
import hashlib
import re

import feedparser
import httpx
from bs4 import BeautifulSoup

from models.schemas import SourceConfig, SourceItem
from collector.collector import candidate_id

USER_AGENT = "ai-engineering-radar/0.1"
MAX_CONTENT_LENGTH = 25_000


def _published_at(entry) -> datetime | None:
    for field in ("published", "updated", "created"):
        value = entry.get(field)
        if not value:
            continue
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError):
            continue
        if parsed.tzinfo is not None:
            return parsed.astimezone(UTC)
    return None


def _text(value: str) -> str:
    return re.sub(r"\s+", " ", BeautifulSoup(value or "", "html.parser").get_text(" ", strip=True)).strip()


def collect_rss(source: SourceConfig, since: datetime) -> list[SourceItem]:
    with httpx.Client(
        timeout=20, follow_redirects=True, headers={"User-Agent": USER_AGENT}
    ) as client:
        response = client.get(str(source.url))
        response.raise_for_status()

    parsed = feedparser.parse(response.content)
    results: list[SourceItem] = []
    for entry in parsed.entries:
        published_at = _published_at(entry)
        if published_at is None or published_at < since.astimezone(UTC):
            continue
        url = entry.get("link", "").strip()
        title = _text(entry.get("title", ""))
        content = _text(
            " ".join(
                [entry.get("summary", "")]
                + [part.get("value", "") for part in entry.get("content", [])]
            )
        )[:MAX_CONTENT_LENGTH]
        if not url or not title:
            continue
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        results.append(
            SourceItem(
                id=candidate_id(url, content_hash),
                source_name=source.name,
                title=title,
                url=url,
                published_at=published_at,
                content=content,
                content_hash=content_hash,
            )
        )
    return results
