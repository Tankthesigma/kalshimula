from src.cache import JsonCache, cache_key


def test_cache_miss_returns_none(tmp_path) -> None:
    cache = JsonCache(tmp_path)

    assert cache.get("nws", {"city": "denver"}) is None


def test_cache_write_then_read_roundtrip(tmp_path) -> None:
    cache = JsonCache(tmp_path)
    payload = {"high_f": 72.5, "source": "nws"}

    cache.set("nws", {"city": "denver", "date": "2025-01-01"}, payload)

    assert cache.get("nws", {"date": "2025-01-01", "city": "denver"}) == payload


def test_cache_key_is_deterministic_for_param_order() -> None:
    left = cache_key("ncei", {"city": "denver", "date": "2025-01-01"})
    right = cache_key("ncei", {"date": "2025-01-01", "city": "denver"})

    assert left == right


def test_cache_key_changes_for_namespace_or_params() -> None:
    key = cache_key("ncei", {"city": "denver"})

    assert key != cache_key("nws", {"city": "denver"})
    assert key != cache_key("ncei", {"city": "chicago"})


def test_cache_write_creates_parent_directories(tmp_path) -> None:
    cache = JsonCache(tmp_path / "nested" / "cache")

    cache.set("power", {"city": "denver"}, {"ok": True})

    assert cache.get("power", {"city": "denver"}) == {"ok": True}
