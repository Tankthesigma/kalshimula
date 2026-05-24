# Tomorrow Model Packet

Model-only decision support. No Kalshi prices, no Kalshi API, no trade instructions.

| city | source | corrected | consensus | delta | promoted | WF MAE | priority | flags |
|---|---|---:|---:|---:|---|---:|---|---|
| nyc | gfs_ens | 64.53 |  |  | True | 1.25 | medium | missing_consensus_delta;large_recent_climate_anomaly |
| chicago | gfs_ens | 64.45 |  |  | True | 1.10 | medium | missing_consensus_delta;large_recent_climate_anomaly |
| miami | gfs_ens | 86.36 |  |  | False | 0.89 | medium | not_contrarian_promoted;missing_consensus_delta |
| austin | gfs_ens | 88.74 |  |  | True | 0.78 | medium | missing_consensus_delta |
| la | gfs_ens | 68.33 |  |  | True | 0.85 | medium | missing_consensus_delta |
| denver | gfs_ens | 64.96 |  |  | False | 1.19 | medium | not_contrarian_promoted;missing_consensus_delta;large_recent_climate_anomaly |
| philadelphia | gfs_ens | 62.25 |  |  | True | 0.98 | medium | missing_consensus_delta;large_recent_climate_anomaly |
| houston | gfs_ens | 92.87 |  |  | True |  | skip | missing_walkforward;missing_consensus_delta |
| phoenix | gfs_ens | 93.49 |  |  | True | 0.77 | medium | missing_consensus_delta;large_recent_climate_anomaly |
| boston | gfs_ens | 61.19 |  |  | True | 1.33 | medium | missing_consensus_delta;large_recent_climate_anomaly |

Use `high` and `medium` rows as manual paper-check candidates only. Bobby's private audit must confirm whether any model disagreement maps to market edge.
