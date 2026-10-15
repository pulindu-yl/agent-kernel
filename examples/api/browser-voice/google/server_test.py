"""Browser voice example tests: the page, and which calls the server accepts. No model or browser needed."""

from types import SimpleNamespace

from server import index, session_for


def call(origin=None, session="abc"):
    headers = {"host": "localhost:8000", **({"origin": origin} if origin else {})}
    return SimpleNamespace(headers=headers, query_params={"session": session} if session else {})


def test_the_page_calls_the_voice_route_and_speaks_the_wire_protocol():
    page = index()

    assert "/voice?session=" in page
    assert all(f'"{kind}"' in page for kind in ("hello", "said", "interrupt", "error"))


def test_a_call_from_this_site_joins_its_own_session():
    assert session_for(call(origin="http://localhost:8000")) == "abc"
    assert session_for(call()) == "abc"  # not a browser


def test_a_call_from_another_site_or_without_a_session_is_refused():
    assert session_for(call(origin="https://evil.example")) is None
    assert session_for(call(origin="http://localhost:8000", session=None)) is None
