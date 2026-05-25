# NBM Historical Feasibility Spike - 2026-05-25

## Verdict

Historical NBM text backfill is feasible from the official NOAA AWS Open Data
bucket, not from the short-retention NOMADS path.

## Evidence

Official sources:

- AWS Open Data registry lists `noaa-nbm-grib2-pds` as the National Blend of
  Models GRIB2 bucket with no-sign-request access:
  https://registry.opendata.aws/noaa-nbm/
- NOAA MDL documents NBM text archive paths in the same bucket:
  https://vlab.noaa.gov/web/mdl/nbm-text-archives

Direct probes from this repo environment:

- `blend.20260525/00/text/blend_nbhtx.t00z` exists.
- `blend.20260524/00/text/blend_nbhtx.t00z` exists.
- `blend.20260523/00/text/blend_nbhtx.t00z` exists.
- `blend.20260518/00/text/blend_nbhtx.t00z` exists.
- `blend.20260501/00/text/blend_nbhtx.t00z` exists.
- Older spot checks also found text products for `20260415`, `20260301`,
  `20260101`, and `20250501`.

The existing NBM text parser was also smoke-tested against AWS for NYC,
target `2026-05-01`, as-of `2026-05-01T12:00:00Z`; it produced a valid
settlement-station guidance row for KNYC.

## Implementation

The live default remains NOMADS. The desk CLIs now accept `--nbm-base-url`, so
historical runs can explicitly use:

```bash
--nbm-base-url https://noaa-nbm-grib2-pds.s3.amazonaws.com
```

This option is wired through:

- `src/nbm_guidance_cli.py`
- `src/weather_desk_cli.py`
- `src/weather_desk_refresh_cli.py`
- `src/weather_desk_schedule_cli.py`
- `src/weather_desk_backfill_cli.py`

## Caveats

- This proves text-product availability and parser compatibility, not a full
  multi-week calibrated weather-quality verdict.
- NCEI TMAX still lags recent dates, so NBM scoring remains settlement-lagged.
- Historical backfills should report empty/missing packets explicitly rather
  than silently aggregating sparse outputs.
