from datetime import date

from src.models.nbm_guidance import build_nbm_guidance_rows
from src.models.station_rules import StationRule


def test_build_nbm_guidance_rows_uses_settlement_station_and_lst_day() -> None:
    rule = StationRule(
        city="nyc",
        platform="kalshi",
        market_type="high",
        settlement_station="KNYC",
        ghcnd_id="GHCND:USW00094728",
        station_name="Central Park",
        timezone="America/New_York",
        lst_offset=-5,
        dst_policy="lst_year_round",
        unit="fahrenheit",
        rounding_rule="nearest_f",
        settlement_source="nws_cli",
        rule_confidence="high",
    )
    nbh_text = """KNYC NBM V4.2 GUIDANCE    5/25/2026  1300 UTC
FHR  01 02 03 04 05 06 07 08 09 10 11 12
TMP  65 66 67 71 73 72 70 68 66 64 62 61
"""
    nbp_text = """KNYC NBM V4.2 GUIDANCE    5/25/2026  1300 UTC
FHR   12 24
TXNP1 68 70
TXNP5 73 75
TXNP9 78 80
"""

    rows = build_nbm_guidance_rows(
        nbh_text=nbh_text,
        nbp_text=nbp_text,
        target=date(2026, 5, 25),
        as_of_ts="2026-05-25T18:00:00Z",
        rules=[rule],
    )

    assert rows["city"].tolist() == ["nyc"]
    assert rows["station_id"].tolist() == ["KNYC"]
    assert rows["guidance_point_f"].tolist() == [73]
    assert rows["guidance_q10_f"].tolist() == [68]
    assert rows["guidance_q50_f"].tolist() == [73]
    assert rows["guidance_q90_f"].tolist() == [78]
    assert rows["valid_ts_utc"].tolist() == ["2026-05-26T01:00:00+00:00"]


def test_build_nbm_guidance_rows_falls_back_to_hourly_max() -> None:
    rule = StationRule(
        city="la",
        platform="kalshi",
        market_type="high",
        settlement_station="KLAX",
        ghcnd_id="GHCND:USW00023174",
        station_name="Los Angeles Intl",
        timezone="America/Los_Angeles",
        lst_offset=-8,
        dst_policy="lst_year_round",
        unit="fahrenheit",
        rounding_rule="nearest_f",
        settlement_source="nws_cli",
        rule_confidence="high",
    )
    nbh_text = """KLAX NBM V4.2 GUIDANCE    5/25/2026  1300 UTC
FHR  01 02 03 04 05 06 07 08 09 10 11 12
TMP  59 60 61 64 66 68 67 65 63 61 60 59
"""

    rows = build_nbm_guidance_rows(
        nbh_text=nbh_text,
        target=date(2026, 5, 25),
        as_of_ts="2026-05-25T18:00:00Z",
        rules=[rule],
    )

    assert rows["city"].tolist() == ["la"]
    assert rows["guidance_point_f"].tolist() == [68]
    assert rows["guidance_q50_f"].tolist() == [68]


def test_build_nbm_guidance_rows_uses_peak_percentile_hour_for_highs() -> None:
    rule = StationRule(
        city="denver",
        platform="kalshi",
        market_type="high",
        settlement_station="KDEN",
        ghcnd_id="GHCND:USW00003017",
        station_name="Denver Intl",
        timezone="America/Denver",
        lst_offset=-7,
        dst_policy="lst_year_round",
        unit="fahrenheit",
        rounding_rule="nearest_f",
        settlement_source="nws_cli",
        rule_confidence="high",
    )
    nbh_text = """KDEN NBM V4.2 GUIDANCE    5/04/2026  0700 UTC
FHR  01 02 03 04 05 06 07 08 09 10 11 12
TMP  48 49 51 55 62 69 73 75 72 65 58 52
"""
    nbp_text = """KDEN NBM V4.2 GUIDANCE    5/04/2026  0700 UTC
FHR   01 02 03 04 05 06 07 08 09 10 11 12
TXNP1 31 33 35 40 49 58 65 67 64 55 48 42
TXNP5 36 38 41 48 58 68 74 76 73 62 54 47
TXNP9 44 47 51 58 68 78 84 86 82 70 61 54
"""

    rows = build_nbm_guidance_rows(
        nbh_text=nbh_text,
        nbp_text=nbp_text,
        target=date(2026, 5, 4),
        as_of_ts="2026-05-04T07:20:00Z",
        rules=[rule],
    )

    assert rows["city"].tolist() == ["denver"]
    assert rows["guidance_point_f"].tolist() == [76]
    assert rows["guidance_q10_f"].tolist() == [67]
    assert rows["guidance_q50_f"].tolist() == [76]
    assert rows["guidance_q90_f"].tolist() == [86]
    assert rows["valid_ts_utc"].tolist() == ["2026-05-04T15:00:00+00:00"]


def test_build_nbm_guidance_rows_parses_nbp_rows_with_leading_spaces_and_day_bars() -> None:
    rule = StationRule(
        city="denver",
        platform="kalshi",
        market_type="high",
        settlement_station="KDEN",
        ghcnd_id="GHCND:USW00003017",
        station_name="Denver Intl",
        timezone="America/Denver",
        lst_offset=-7,
        dst_policy="lst_year_round",
        unit="fahrenheit",
        rounding_rule="nearest_f",
        settlement_source="nws_cli",
        rule_confidence="high",
    )
    nbh_text = """KDEN NBM V4.2 GUIDANCE    5/04/2026  1900 UTC
UTC  20 21 22 23 00 01
TMP  70 69 67 64 62 59
"""
    nbp_text = """KDEN    NBM V5.0 NBP GUIDANCE    5/04/2026  1900 UTC
    TUE 05| WED 06| THU 07|
 UTC    12| 00  12| 00  12|
 FHR    17| 29  41| 53  65|221
 TXNP1  31| 36  19| 29  13| 74
 TXNP5  36| 40  22| 32  19| 84
 TXNP9  39| 46  29| 37  23| 90
"""

    rows = build_nbm_guidance_rows(
        nbh_text=nbh_text,
        nbp_text=nbp_text,
        target=date(2026, 5, 5),
        as_of_ts="2026-05-04T19:20:00Z",
        rules=[rule],
    )

    assert rows["city"].tolist() == ["denver"]
    assert rows["guidance_point_f"].tolist() == [40]
    assert rows["guidance_q10_f"].tolist() == [36]
    assert rows["guidance_q50_f"].tolist() == [40]
    assert rows["guidance_q90_f"].tolist() == [46]
    assert rows["valid_ts_utc"].tolist() == ["2026-05-06T00:00:00+00:00"]


def test_build_nbm_guidance_rows_parses_packed_three_digit_temperatures() -> None:
    rule = StationRule(
        city="phoenix",
        platform="kalshi",
        market_type="high",
        settlement_station="KPHX",
        ghcnd_id="GHCND:USW00023183",
        station_name="Phoenix Sky Harbor",
        timezone="America/Phoenix",
        lst_offset=-7,
        dst_policy="no_dst",
        unit="fahrenheit",
        rounding_rule="nearest_f",
        settlement_source="nws_cli",
        rule_confidence="high",
    )
    nbh_text = """KPHX NBM V5.0 NBH GUIDANCE    5/09/2026  1400 UTC
UTC  15 16 17 18 19 20 21 22 23 00 01 02
TMP  81 86 91 94 97100102103103102101 99
"""

    rows = build_nbm_guidance_rows(
        nbh_text=nbh_text,
        target=date(2026, 5, 9),
        as_of_ts="2026-05-09T14:20:00Z",
        rules=[rule],
    )

    assert rows["city"].tolist() == ["phoenix"]
    assert rows["guidance_point_f"].tolist() == [103.0]
    assert rows["guidance_q50_f"].tolist() == [103.0]


def test_build_nbm_guidance_rows_parses_packed_values_after_two_digit_start() -> None:
    rule = StationRule(
        city="phoenix",
        platform="kalshi",
        market_type="high",
        settlement_station="KPHX",
        ghcnd_id="GHCND:USW00023183",
        station_name="Phoenix Sky Harbor",
        timezone="America/Phoenix",
        lst_offset=-7,
        dst_policy="no_dst",
        unit="fahrenheit",
        rounding_rule="nearest_f",
        settlement_source="nws_cli",
        rule_confidence="high",
    )
    nbh_text = """KPHX NBM V5.0 NBH GUIDANCE    5/10/2026  1700 UTC
UTC  18 19 20 21 22 23 00 01 02 03 04 05
TMP  97100102104105105104104100 96 93 90
"""

    rows = build_nbm_guidance_rows(
        nbh_text=nbh_text,
        target=date(2026, 5, 10),
        as_of_ts="2026-05-10T17:20:00Z",
        rules=[rule],
    )

    assert rows["city"].tolist() == ["phoenix"]
    assert rows["guidance_point_f"].tolist() == [105.0]
    assert rows["guidance_q50_f"].tolist() == [105.0]
