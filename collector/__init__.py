from collector.collector import (
    candidate_id,
    canonicalize_url,
    collect_sources,
)
from collector.rss import collect_rss
from collector.webpage import collect_webpage

__all__ = [
    "candidate_id",
    "canonicalize_url",
    "collect_rss",
    "collect_sources",
    "collect_webpage",
]
