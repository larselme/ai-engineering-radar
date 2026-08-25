from datetime import UTC, datetime
import hashlib
import json
import re
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from collector.collector import candidate_id
from models.schemas import SourceConfig, SourceItem

USER_AGENT = "ai-engineering-radar/0.1"
MAX_CONTENT_LENGTH = 25_000


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _jsonld_date(soup: BeautifulSoup) -> datetime | None:
    for node in soup.select('script[type="application/ld+json"]'):
        try:
            data = json.loads(node.string or node.get_text())
        except (TypeError, json.JSONDecodeError):
            continue
        values = data if isinstance(data, list) else [data]
        pending = list(values)
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                if date := _parse_date(value.get("datePublished")):
                    return date
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
    return None


def _text(element) -> str:
    return re.sub(r"\s+", " ", element.get_text(" ", strip=True)).strip()


def collect_webpage(source: SourceConfig, since: datetime) -> list[SourceItem]:
    with httpx.Client(
        timeout=20, follow_redirects=True, headers={"User-Agent": USER_AGENT}
    ) as client:
        index_response = client.get(str(source.url))
        index_response.raise_for_status()
        index_soup = BeautifulSoup(index_response.text, "html.parser")
        links: list[tuple[str, datetime | None]] = []
        for anchor in index_soup.select("a[href]"):
            href = anchor.get("href")
            if not href or href.strip().lower() == "none":
                continue
            url = urljoin(str(source.url), href)
            if not any(url.startswith(prefix) for prefix in source.article_url_prefixes):
                continue
            date = None
            parent = anchor.parent
            while parent is not None and parent.name not in {"body", "html"}:
                time = parent.select_one("time[datetime]")
                if time is not None and len(parent.select("a[href]")) == 1:
                    date = _parse_date(time.get("datetime"))
                    break
                parent = parent.parent
            links.append((url, date))

        results: list[SourceItem] = []
        for url, index_date in links:
            article_response = client.get(url)
            article_response.raise_for_status()
            article_soup = BeautifulSoup(article_response.text, "html.parser")
            published_at = index_date
            if published_at is None:
                time = article_soup.select_one("time[datetime]")
                published_at = _parse_date(time.get("datetime")) if time else None
            if published_at is None:
                published_at = _jsonld_date(article_soup)
            if published_at is None or published_at < since.astimezone(UTC):
                continue
            body = article_soup.select_one("article") or article_soup.select_one("main")
            content = _text(body or article_soup)[:MAX_CONTENT_LENGTH]
            title_node = article_soup.select_one("h1") or article_soup.select_one("title")
            title = _text(title_node) if title_node else url
            if source.include_title_terms and not any(
                term.lower() in title.lower() for term in source.include_title_terms
            ):
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
