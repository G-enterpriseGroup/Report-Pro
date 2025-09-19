# updater.py
import os, re, json, math, time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import gspread
from gspread_dataframe import set_with_dataframe
from google.oauth2.service_account import Credentials
from google.auth.exceptions import TransportError

# ========= Config via GitHub Actions secrets =========
SHEET_URL       = os.environ["SHEET_URL"].strip()                    # required
WORKSHEET_NAME  = os.environ.get("WORKSHEET_NAME", "Copy").strip()   # optional

# ========= Cell anchors =========
CELL_TICKER = "A2"
CELL_OCC    = "A3"

OUT_OCC_HEADER_CELL   = "A4"
OUT_OCC_VALUES_CELL   = "A5"
OUT_EXPS_HEADER_CELL  = "A7"
OUT_EXPS_START_CELL   = "A8"
OUT_PUTS_HEADER_CELL  = "C7"

# ========= Google auth (Service Account JSON in GOOGLE_CREDENTIALS) =========
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
creds_info = json.loads(os.environ["GOOGLE_CREDENTIALS"])
creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
gc = gspread.authorize(creds)

# ========= Utilities =========
def _sheet_id(url_or_id: str) -> str:
    s = url_or_id.strip().strip('"').strip("'")
    m = re.search(r"/spreadsheets/d/([A-Za-z0-9-_]+)", s)
    if m: return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9-_]{20,}", s): return s
    raise ValueError(f"Could not parse Sheet ID from SHEET_URL='{s[:80]}'")

def a1_to_rowcol(a1: str) -> Tuple[int, int]:
    col_str = ''.join(filter(str.isalpha, a1))
    row_str = ''.join(filter(str.isdigit, a1))
    row = int(row_str); col = 0
    for i, c in enumerate(reversed(col_str.upper())):
        col += (ord(c) - 64) * (26 ** i)
    return row, col

def json_safe(x: Any) -> Any:
    """
    Convert values to JSON/Sheets-safe primitives.
    NaN/Inf -> ""; numpy scalars -> Python scalars; pandas NA -> "".
    """
    if x is None:
        return ""
    if isinstance(x, (np.generic,)):
        x = np.asarray(x).item()
    if isinstance(x, float):
        return x if math.isfinite(x) else ""
    # Handle pandas NA types
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return x

def df_json_safe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.replace([np.inf, -np.inf], pd.NA).fillna("")
    # Cast to object so gspread doesn't see numpy dtypes
    return out.astype(object)

def with_retries(fn, *args, retries: int = 3, delay: float = 0.8, **kwargs):
    """
    Retry wrapper for transient network errors / API hiccups.
    """
    last_ex = None
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except (TransportError, gspread.exceptions.APIError, ConnectionError) as e:
            last_ex = e
            time.sleep(delay * (1.5 ** i))
        except Exception as e:
            # Non-transient: bubble up
            raise
    if last_ex:
        raise last_ex

# ========= Domain helpers =========
def parse_occ(contract: str) -> Optional[Dict[str, Any]]:
    m = re.match(r"^([A-Za-z]+)(\d{2})(\d{2})(\d{2})([CP])(\d{8})$", (contract or "").strip())
    if not m: return None
    und, yy, mm, dd, cp, strike_code = m.groups()
    year  = 2000 + int(yy); month = int(mm); day = int(dd)
    expiry_iso = f"{year:04d}-{month:02d}-{day:02d}"
    strike = int(strike_code) / 1000.0
    return {"underlying": und.upper(), "type": cp, "strike": strike, "expiry_iso": expiry_iso}

def _fmt_mid(bid, ask):
    try:
        if pd.notna(bid) and pd.notna(ask):
            b = float(bid); a = float(ask)
            if b > 0 and a > 0:
                return round((b + a) / 2, 4)
    except Exception:
        pass
    return ""

def normalize_puts(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    required = ["contractSymbol","strike","lastPrice","bid","ask","openInterest","impliedVolatility","volume","inTheMoney"]
    for col in required:
        if col not in out.columns:
            out[col] = pd.NA
    out["mid"] = out.apply(lambda r: _fmt_mid(r.get("bid"), r.get("ask")), axis=1)
    out = out.rename(columns={
        "lastPrice": "last",
        "impliedVolatility": "impliedVol",
        "openInterest": "openInterest",
    })[["contractSymbol","strike","last","bid","ask","mid","openInterest","impliedVol","volume","inTheMoney"]]
    return df_json_safe(out)

def lookup_occ_with_yf(occ: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    meta = parse_occ(occ)
    if not meta:
        return None, "Invalid OCC contract"

    t = yf.Ticker(meta["underlying"])
    expiries = list(getattr(t, "options", []) or [])
    if not expiries:
        return None, f"No expirations for {meta['underlying']}"

    search_order = [meta["expiry_iso"]] + [e for e in expiries if e != meta["expiry_iso"]] if meta["expiry_iso"] in expiries else expiries
    for exp in search_order:
        try:
            chain = with_retries(t.option_chain, exp, retries=2, delay=0.6)
        except Exception:
            continue
        pool = chain.puts if meta["type"] == "P" else chain.calls
        exact = pool[pool.get("contractSymbol") == occ] if "contractSymbol" in pool.columns else pd.DataFrame()
        if not exact.empty:
            r = exact.iloc[0].to_dict()
            row = {
                "contractSymbol": r.get("contractSymbol",""),
                "underlying": meta["underlying"],
                "expiry": exp,
                "type": ("PUT" if meta["type"]=="P" else "CALL"),
                "strike": json_safe(r.get("strike", meta["strike"])),
                "last": json_safe(r.get("lastPrice","")),
                "bid": json_safe(r.get("bid","")),
                "ask": json_safe(r.get("ask","")),
                "mid": json_safe(_fmt_mid(r.get("bid"), r.get("ask"))),
                "oi": json_safe(r.get("openInterest","")),
                "iv": json_safe(r.get("impliedVolatility","")),
                "volume": json_safe(r.get("volume","")),
                "itm": json_safe(r.get("inTheMoney","")),
                "currency": "USD",
            }
            # Ensure all are JSON-safe primitives
            row = {k: json_safe(v) for k, v in row.items()}
            return row, None
    return None, "Contract not found"

def safe_ws_update(ws, range_name: str, values):
    """
    gspread update with named args + JSON-safe conversion + retry.
    """
    # Convert any nested values to JSON-safe
    safe_values = []
    for row in values:
        safe_values.append([json_safe(v) for v in row])
    return with_retries(ws.update, range_name=range_name, values=safe_values, retries=3, delay=0.7)

# ========= Main =========
def main():
    sheet_id = _sheet_id(SHEET_URL)
    sh = with_retries(gc.open_by_key, sheet_id, retries=3, delay=0.7)
    ws = with_retries(sh.worksheet, WORKSHEET_NAME, retries=3, delay=0.7)

    # Inputs
    ticker = (ws.acell(CELL_TICKER).value or "").strip().upper()
    occ    = (ws.acell(CELL_OCC).value or "").strip().upper()

    # OCC snapshot header
    headers = [["ContractSymbol","Underlying","Expiry","Type","Strike","Last","Bid","Ask","Mid","OpenInterest","ImpliedVol","Volume","InTheMoney","Currency"]]
    safe_ws_update(ws, OUT_OCC_HEADER_CELL, headers)

    # OCC snapshot value
    if occ:
        row, err = lookup_occ_with_yf(occ)
        if row:
            safe_ws_update(ws, OUT_OCC_VALUES_CELL, [[
                row["contractSymbol"], row["underlying"], row["expiry"], row["type"], row["strike"],
                row["last"], row["bid"], row["ask"], row["mid"], row["oi"], row["iv"], row["volume"], row["itm"], row["currency"]
            ]])
        else:
            safe_ws_update(ws, OUT_OCC_VALUES_CELL, [[err]])
    else:
        safe_ws_update(ws, OUT_OCC_VALUES_CELL, [["(Put OCC in A3)"]])

    # Expirations + nearest puts for ticker
    if ticker:
        t = yf.Ticker(ticker)
        expiries = list(getattr(t, "options", []) or [])
        safe_ws_update(ws, OUT_EXPS_HEADER_CELL, [["Expirations (ISO)"]])
        if expiries:
            # Write expirations list
            safe_ws_update(ws, OUT_EXPS_START_CELL, [[d] for d in expiries])

            # Choose earliest (sorted lexicographically works for ISO yyyy-mm-dd)
            nearest = sorted(expiries)[0]

            # Fetch chain (retry)
            chain = with_retries(t.option_chain, nearest, retries=2, delay=0.6)
            puts_df = normalize_puts(chain.puts)

            # Section header
            safe_ws_update(ws, OUT_PUTS_HEADER_CELL, [[f"Puts @ {nearest}"]])

            # Write DF below header
            r, c = a1_to_rowcol(OUT_PUTS_HEADER_CELL)
            puts_df = df_json_safe(puts_df)
            # gspread_dataframe handles conversion, but we already sanitized
            with_retries(set_with_dataframe, ws, puts_df, row=r+1, col=c, retries=2, delay=0.6)
        else:
            safe_ws_update(ws, OUT_EXPS_START_CELL, [["No expirations available"]])
    else:
        safe_ws_update(ws, OUT_EXPS_HEADER_CELL, [["(Put Ticker in A2)"]])

    print("Done (puts only).")

if __name__ == "__main__":
    main()
