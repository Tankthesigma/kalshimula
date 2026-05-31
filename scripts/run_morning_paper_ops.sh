#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

TARGET_DATE="${TARGET_DATE:-$(TZ=America/Chicago date +%F)}"
STAMP="${STAMP:-$(TZ=America/Chicago date +%H%M%S)}"
MODEL_RUN_DIR="${MODEL_RUN_DIR:-data/runs/may2024_apr2026_10city_openmeteo_sources_2yr}"
SOURCE_REPO_ROOT="${SOURCE_REPO_ROOT:-/mnt/c/Users/vasud/OneDrive/Documents/kalshimula/kalshimula-model-longrun}"
DECISION_TIME_LABEL="${DECISION_TIME_LABEL:-}"
# Operational (Bobby handoff) scope = the traded-high set ONLY (10 cities). The model run's
# recommended_sources.csv selects a source for exactly these 10 (all gfs_ens); the other 10
# research cities have NO selected source, so each one fans out to all 7 Open-Meteo models
# (predict._fetch_all_parallel) — that fanout, not the 10 traded gfs_ens requests, is what
# saturates the ensemble endpoint and 429-collapses the morning run. Research/backfill paths
# can still override CITIES explicitly for the full 20-city sweep.
CITIES="${CITIES:-nyc,chicago,miami,austin,la,denver,philadelphia,houston,phoenix,boston}"
OUT_ROOT="${OUT_ROOT:-outputs/private_pink_sheets/${TARGET_DATE}/${STAMP}}"
BRIDGE_ENV="${BRIDGE_ENV:-/mnt/c/Users/vasud/OneDrive/Documents/discord-agent-bridge-wsl/.env}"
BRIDGE_CLI="${BRIDGE_CLI:-/mnt/c/Users/vasud/OneDrive/Documents/discord-agent-bridge-wsl/bridge/discord_mailbox.py}"

if [[ "${MODEL_RUN_DIR}" != /* ]]; then
  MODEL_RUN_DIR="${REPO_ROOT}/${MODEL_RUN_DIR}"
fi
if [[ ! -d "${MODEL_RUN_DIR}" ]]; then
  FALLBACK_MODEL_RUN_DIR="${SOURCE_REPO_ROOT}/data/runs/may2024_apr2026_10city_openmeteo_sources_2yr"
  if [[ -d "${FALLBACK_MODEL_RUN_DIR}" ]]; then
    MODEL_RUN_DIR="${FALLBACK_MODEL_RUN_DIR}"
  fi
fi
if [[ ! -d "${MODEL_RUN_DIR}" ]]; then
  echo "missing model run dir: ${MODEL_RUN_DIR}"
  exit 1
fi

if [[ -z "${DECISION_TIME_LABEL}" ]]; then
  CT_HOUR="$(TZ=America/Chicago date +%H)"
  CT_MINUTE="$(TZ=America/Chicago date +%M)"
  if [[ "${CT_MINUTE}" -ge 55 ]]; then
    DECISION_TIME_LABEL="$(TZ=America/Chicago date -d '+1 hour' +%H)"
  else
    DECISION_TIME_LABEL="${CT_HOUR}"
  fi
fi

mkdir -p "${OUT_ROOT}"

LOCK_FILE="${LOCK_FILE:-outputs/private_pink_sheets/morning_ops.lock}"
mkdir -p "$(dirname "${LOCK_FILE}")"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "morning paper ops already running; skipping ${TARGET_DATE}/${STAMP}"
  exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-/home/vasud/miniconda3/bin/python3}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

"${PYTHON_BIN}" - <<'PY'
import pandas  # noqa: F401
PY

"${PYTHON_BIN}" -m src.weather_desk_refresh_cli \
  --model-run-dir "${MODEL_RUN_DIR}" \
  --cities "${CITIES}" \
  --date "${TARGET_DATE}" \
  --threshold-offsets=-6,-4,-2,0,2,4,6 \
  --multi-source-mode single \
  --station-rules config/station_rule_table.csv \
  --market-type high \
  --decision-time-label "${DECISION_TIME_LABEL}" \
  --observation-store "${OUT_ROOT}/asos_store.csv" \
  --fetch-live \
  --update-observation-store \
  --include-nws-guidance \
  --include-nbm-guidance \
  --no-require-gate \
  --allow-source-fallback \
  --out-dir "${OUT_ROOT}/weather_packet"

CSV="${OUT_ROOT}/weather_packet/weather_desk/weather_analyst/weather_analyst_packet.csv"
SUMMARY="${OUT_ROOT}/discord_summary.txt"
REFRESH_MANIFEST="${OUT_ROOT}/weather_packet/weather_desk_refresh_manifest.json"

"${PYTHON_BIN}" - "$CSV" "$SUMMARY" "$OUT_ROOT" "$REFRESH_MANIFEST" <<'PY'
import csv
import json
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
out_root = Path(sys.argv[3])
manifest_path = Path(sys.argv[4])

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
git_commit = manifest.get("git_commit") or "unknown"
as_of_ts = manifest.get("as_of_ts_utc") or "unknown"
manifest_notes = manifest.get("notes") or []
nbm_note = next(
    (note for note in manifest_notes if "NBM guidance unavailable" in str(note)),
    None,
)

rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
rank = {"clean": 0, "review": 1, "veto": 2}
rows.sort(
    key=lambda r: (
        rank.get((r.get("desk_priority") or "").lower(), 9),
        -float(r.get("top_bin_probability") or 0),
        r.get("city") or "",
    )
)

counts: dict[str, int] = {}
for row in rows:
    counts[row.get("desk_priority") or "unknown"] = counts.get(row.get("desk_priority") or "unknown", 0) + 1

lines = [
    "CODEX WEATHER TARGETS READY",
    "Label: WEATHER-ONLY / PAPER-ONLY / NOT REAL MONEY.",
    "This is NOT the trade sheet. Bobby-private must add executable Kalshi odds, $10 economics, and the final PAPER TRACK picks.",
    f"Output: {out_root}",
    f"Promotable clean subset: {out_root}/weather_packet/weather_desk/weather_analyst/weather_analyst_clean_rows.csv",
    f"Mainline git: {git_commit} | as_of_utc: {as_of_ts}",
    f"Priority counts: {counts}",
    (
        "Calibration coverage: "
        f"{sum('uncalibrated_source_policy' in (row.get('risk_flags') or '') for row in rows)} "
        "rows excluded from clean promotion for missing bias/interval coverage."
    ),
    "",
    "CODEX CLEAN WEATHER TARGETS FOR BOBBY TO PRICE:",
]

if nbm_note:
    lines.extend(
        [
            f"NBM status: {nbm_note}",
            "",
        ]
    )

usable = [
    r for r in rows
    if (r.get("desk_priority") or "").lower() == "clean"
]
if not usable:
    lines.extend([
        "- NONE: no clean rows passed the weather gate.",
        "- NO PAPER SHORT LIST TODAY. Do not manufacture picks from this packet.",
    ])
else:
    for idx, row in enumerate(usable, start=1):
        delta = row.get("model_minus_nws_f") or "NA"
        flags = row.get("risk_flags") or ""
        lines.append(
            "{idx}. {city} {market}: source={source} calibrated={calibrated} "
            "target top={top} ({prob:.0%}), point={point:.1f}, q10-q90={q10:.0f}-{q90:.0f}, "
            "priority={priority}, NWS_delta={delta}, flags={flags}".format(
                idx=idx,
                city=row.get("city"),
                market=row.get("market_type"),
                source=row.get("source_policy"),
                calibrated=row.get("calibration_supported"),
                top=row.get("top_bin_label"),
                prob=float(row.get("top_bin_probability") or 0),
                point=float(row.get("point_f") or 0),
                q10=float(row.get("q10_f") or 0),
                q90=float(row.get("q90_f") or 0),
                priority=row.get("desk_priority"),
                delta=delta,
                flags=flags,
            )
        )

lines.extend([
    "",
    "Instruction to Bobby-private: price only clean rows for any short paper list, then attach the full board.",
    "If there are zero clean rows, post no short list at all.",
    "Rows marked review or veto are DO-NOT-TRADE weather audit rows. Do not promote them into picks or a short list.",
    "Do not call this an edge unless executable $10 economics and the frozen ranking rule are shown.",
    "",
    "Full city board:",
])

for row in rows:
    delta = row.get("model_minus_nws_f") or "NA"
    flags = row.get("risk_flags") or ""
    lines.append(
        "- {city} {market}: {priority} source={source} calibrated={calibrated} "
        "point={point:.1f} q10-q90={q10:.0f}-{q90:.0f} top={top} ({prob:.0%}) "
        "NWS_delta={delta} flags={flags}".format(
            city=row.get("city"),
            market=row.get("market_type"),
            priority=row.get("desk_priority"),
            source=row.get("source_policy"),
            calibrated=row.get("calibration_supported"),
            point=float(row.get("point_f") or 0),
            q10=float(row.get("q10_f") or 0),
            q90=float(row.get("q90_f") or 0),
            top=row.get("top_bin_label"),
            prob=float(row.get("top_bin_probability") or 0),
            delta=delta,
            flags=flags,
        )
    )

lines.extend(
    [
        "",
        "Operational rule: scheduled rows only count if generated before the entry window; late screenshots are discretionary.",
    ]
)
summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_path)
PY

cat "${SUMMARY}"

if [[ -f "${BRIDGE_ENV}" && -f "${BRIDGE_CLI}" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "${BRIDGE_ENV}"
  set +a
  "${PYTHON_BIN}" "${BRIDGE_CLI}" reply "$(cat "${SUMMARY}")"
fi
