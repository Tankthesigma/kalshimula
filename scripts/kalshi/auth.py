"""Kalshi auth helper. Read-only, no trading."""
from __future__ import annotations
import base64, json, time, urllib.parse, urllib.request
from pathlib import Path
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

KEY_ID = "115d972c-0cfa-44c5-a322-f722d981bef0"
PEM_PATH = Path(__file__).resolve().parent.parent.parent / ".secrets" / "kalshi.pem"
BASE = "https://api.elections.kalshi.com/trade-api/v2"  # current public host
ALT = "https://trading-api.kalshi.com/trade-api/v2"     # legacy host

def _load_key():
    with open(PEM_PATH, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)

def _sign(ts_ms: str, method: str, path: str) -> str:
    msg = (ts_ms + method + path).encode("utf-8")
    sig = _load_key().sign(
        msg,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode("ascii")

def get(path: str, params: dict | None = None, host: str = BASE) -> tuple[int, object]:
    if params:
        qs = urllib.parse.urlencode(params)
        url = f"{host}{path}?{qs}"
        sign_path = f"/trade-api/v2{path}"  # signature uses path without query
    else:
        url = f"{host}{path}"
        sign_path = f"/trade-api/v2{path}"
    ts_ms = str(int(time.time() * 1000))
    sig = _sign(ts_ms, "GET", sign_path)
    headers = {
        "KALSHI-ACCESS-KEY": KEY_ID,
        "KALSHI-ACCESS-TIMESTAMP": ts_ms,
        "KALSHI-ACCESS-SIGNATURE": sig,
        "Accept": "application/json",
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        return e.code, body

if __name__ == "__main__":
    # quick probe
    for host_label, host in (("elections", BASE), ("legacy", ALT)):
        status, body = get("/exchange/status", host=host)
        print(f"[{host_label}] /exchange/status -> {status}")
        if status == 200:
            print(json.dumps(body, indent=2)[:200])
        else:
            print(str(body)[:200])
