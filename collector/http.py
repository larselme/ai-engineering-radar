import ssl

import httpx

USER_AGENT = "ai-engineering-radar/0.1"


def client_kwargs() -> dict:
    kwargs = {
        "timeout": 20,
        "follow_redirects": True,
        "headers": {"User-Agent": USER_AGENT},
    }
    try:
        import truststore

        kwargs["verify"] = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        pass
    return kwargs


def create_client() -> httpx.Client:
    return httpx.Client(**client_kwargs())