from datetime import UTC, datetime
from pathlib import Path
import ssl
import sys
import types

import pytest
from collector.collector import candidate_id, canonicalize_url, collect_sources
from collector.http import client_kwargs
from collector.rss import collect_rss
from collector.webpage import collect_webpage
from models.schemas import SourceConfig, SourceKind, SourceItem

FIXTURES = Path(__file__).parent / "fixtures"


class FakeResponse:
    def __init__(self, body: str, status_code: int = 200):
        self.content = body.encode()
        self.text = body
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeClient:
    responses = {}

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def get(self, url):
        response = self.responses.get(url)
        if isinstance(response, Exception):
            raise response
        if response is None:
            raise RuntimeError(f"missing fixture for {url}")
        return response


def test_http_client_uses_truststore_when_available(monkeypatch):
    fake_truststore = types.SimpleNamespace(
        SSLContext=lambda protocol: ("truststore-context", protocol)
    )
    monkeypatch.setitem(sys.modules, "truststore", fake_truststore)

    kwargs = client_kwargs()

    assert kwargs["verify"] == ("truststore-context", ssl.PROTOCOL_TLS_CLIENT)


def test_http_client_omits_verify_override_without_truststore(monkeypatch):
    monkeypatch.delitem(sys.modules, "truststore", raising=False)
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "truststore":
            raise ImportError("missing truststore")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    kwargs = client_kwargs()

    assert "verify" not in kwargs


def source(kind, url="https://example.com/index", **kwargs):
    return SourceConfig(name="Example", kind=kind, url=url, **kwargs)


def test_rss_date_filtering_and_utc_normalization(monkeypatch):
    from collector import rss
    FakeClient.responses = {"https://example.com/feed": FakeResponse((FIXTURES / "sample_feed.xml").read_text())}
    monkeypatch.setattr(rss, "create_client", lambda: FakeClient())
    items = collect_rss(source(SourceKind.RSS, "https://example.com/feed"), datetime(2026, 8, 25, tzinfo=UTC))
    assert [item.title for item in items] == ["Fresh"]
    assert items[0].published_at == datetime(2026, 8, 25, 8, tzinfo=UTC)


def test_webpage_extracts_articles_and_index_time(monkeypatch):
    from collector import webpage
    base = "https://example.com/index"
    FakeClient.responses = {
        base: FakeResponse((FIXTURES / "sample_index.html").read_text()),
        "https://example.com/articles/with-time": FakeResponse("<article><h1>Title</h1><p>Body</p></article>"),
        "https://example.com/articles/jsonld": FakeResponse(
            '<script type="application/ld+json">{"datePublished":"2026-08-25T09:00:00Z"}</script>'
            "<main><h1>JSON title</h1>long text</main>"
        ),
    }
    monkeypatch.setattr(webpage, "create_client", lambda: FakeClient())
    items = collect_webpage(source(SourceKind.WEBPAGE, base, article_url_prefixes=["https://example.com/articles/"]), datetime(2026, 8, 25, tzinfo=UTC))
    assert len(items) == 2
    assert items[0].published_at == datetime(2026, 8, 25, 8, tzinfo=UTC)
    assert items[1].published_at == datetime(2026, 8, 25, 9, tzinfo=UTC)


def test_webpage_fallbacks_to_json_ld_date(monkeypatch):
    from collector import webpage
    base = "https://example.com/index"
    FakeClient.responses = {
        base: FakeResponse('<a href="/articles/jsonld">JSON-LD</a>'),
        "https://example.com/articles/jsonld": FakeResponse(
            '<script type="application/ld+json">{"datePublished":"2026-08-25T09:00:00Z"}</script>'
            "<main>Body</main>"
        ),
    }
    monkeypatch.setattr(webpage, "create_client", lambda: FakeClient())
    items = collect_webpage(source(SourceKind.WEBPAGE, base, article_url_prefixes=["https://example.com/articles/"]), datetime(2026, 8, 25, tzinfo=UTC))
    assert items[0].published_at == datetime(2026, 8, 25, 9, tzinfo=UTC)


def test_content_is_capped(monkeypatch):
    from collector import webpage
    base = "https://example.com/index"
    FakeClient.responses = {
        base: FakeResponse('<a href="/articles/long">Long</a><time datetime="2026-08-25T00:00:00Z"></time>'),
        "https://example.com/articles/long": FakeResponse("<article>" + ("x" * 30000) + "</article>"),
    }
    monkeypatch.setattr(webpage, "create_client", lambda: FakeClient())
    assert len(collect_webpage(source(SourceKind.WEBPAGE, base, article_url_prefixes=["https://example.com/articles/"]), datetime(2026, 8, 24, tzinfo=UTC))[0].content) == 25000


def test_url_canonicalization_and_candidate_id():
    url = "https://example.com/a?keep=1&utm_source=test#section"
    assert canonicalize_url(url) == "https://example.com/a?keep=1"
    assert candidate_id(url, "hash") == candidate_id("https://example.com/a?keep=1", "other")


def test_canonicalization_removes_all_common_tracking_parameters():
    url = "https://example.com/a?utm_source=x&keep=1&utm_medium=y&utm_campaign=z&utm_term=t&utm_content=c"
    assert canonicalize_url(url) == "https://example.com/a?keep=1"


def test_duplicate_urls_collapse(monkeypatch):
    items = [
        SourceItem(id="a", source_name="A", title="A", url="https://example.com/a?utm_source=x#one", published_at=datetime.now(UTC), content="x", content_hash="h1"),
        SourceItem(id="b", source_name="A", title="B", url="https://example.com/a", published_at=datetime.now(UTC), content="y", content_hash="h2"),
    ]
    import collector.rss as rss
    monkeypatch.setattr(rss, "collect_rss", lambda s, since: items)
    result, errors = collect_sources([source(SourceKind.RSS, "https://example.com/feed")], datetime.now(UTC))
    assert len(result) == 1 and not errors


def test_different_urls_with_same_content_hash_survive(monkeypatch):
    items = [
        SourceItem(id="a", source_name="A", title="A", url="https://example.com/a", published_at=datetime.now(UTC), content="same", content_hash="same"),
        SourceItem(id="b", source_name="A", title="B", url="https://example.com/b", published_at=datetime.now(UTC), content="same", content_hash="same"),
    ]
    import collector.rss as rss
    monkeypatch.setattr(rss, "collect_rss", lambda s, since: items)
    result, errors = collect_sources([source(SourceKind.RSS, "https://example.com/feed")], datetime.now(UTC))
    assert result == items
    assert not errors


def test_one_source_failure_is_reported_and_other_results_survive(monkeypatch):
    import collector.rss as rss
    good = SourceItem(id="a", source_name="Good", title="A", url="https://example.com/a", published_at=datetime.now(UTC), content="x", content_hash="h")
    def fake(source_config, since):
        if source_config.name == "Bad":
            raise OSError("offline")
        return [good]
    monkeypatch.setattr(rss, "collect_rss", fake)
    result, errors = collect_sources([source(SourceKind.RSS, "https://example.com/good"), SourceConfig(name="Bad", kind=SourceKind.RSS, url="https://example.com/bad")], datetime.now(UTC))
    assert result == [good]
    assert "Bad" in errors and "offline" in errors["Bad"]
