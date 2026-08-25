import hashlib
from datetime import datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from models.schemas import SourceConfig, SourceItem, SourceKind

TRACKING_PARAMETERS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
}


def canonicalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    query = urlencode(
        [(key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)
         if key.lower() not in TRACKING_PARAMETERS]
    )
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, ""))


def candidate_id(url: str, content_hash: str) -> str:
    identity = canonicalize_url(url) if url and url.strip() else content_hash
    if not identity:
        raise ValueError("candidate identity requires a URL or content hash")
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def collect_sources(
    sources: list[SourceConfig],
    since: datetime,
) -> tuple[list[SourceItem], dict[str, str]]:
    items: list[SourceItem] = []
    source_errors: dict[str, str] = {}
    for source in sources:
        try:
            if source.kind is SourceKind.RSS:
                from collector.rss import collect_rss

                collected = collect_rss(source, since)
            else:
                from collector.webpage import collect_webpage

                collected = collect_webpage(source, since)
            items.extend(collected)
        except Exception as exc:
            source_errors[source.name] = f"{type(exc).__name__}: {exc}"

    unique: list[SourceItem] = []
    seen_urls: set[str] = set()
    seen_content: set[str] = set()
    for item in items:
        url = canonicalize_url(str(item.url))
        if url in seen_urls or item.content_hash in seen_content:
            continue
        seen_urls.add(url)
        seen_content.add(item.content_hash)
        unique.append(item)
    return unique, source_errors
