import pytest

from src.config import Station, get_station, load_stations


def test_load_stations_returns_station_objects() -> None:
    stations = load_stations()

    assert stations
    assert all(isinstance(station, Station) for station in stations.values())


def test_get_station_returns_configured_station() -> None:
    slug = next(iter(load_stations()))

    station = get_station(slug)

    assert station.slug == slug
    assert station.ghcnd_bare == station.ghcnd_id.removeprefix("GHCND:")


def test_get_station_rejects_unknown_slug() -> None:
    with pytest.raises(KeyError):
        get_station("__missing_city__")
