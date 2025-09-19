# updater.py
import os, re, json, math, time
from typing import Any, Dict, Optional, Tuple
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
import gspread
from gspread_dataframe import set_with_dataframe
from google.oauth2.service_account import Credentials
from google.auth.exceptions import TransportError

# ---- OPTIONAL but recommended for theming ----
try:
    from gspread_formatting import (
        CellFormat, Color, TextFormat, NumberFormat,
        format_cell_range, set_frozen, add_banding, BandingTheme, GridRange
    )
    _HAS_FMT = True
except Exception:
    _HAS_FMT = False

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
OUT_PUTS_HEADER_CELL  = "C7"   # Title cell for puts section (table starts on next row)

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

def col_to_a1(col_index: int) -> str:
    # 1 -> A, 2 -> B ...
    s = ""
    n = col_index
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s

def json_safe(x: Any) -> Any:
    """Convert to JSON/Sheets-safe primitive."""
    if x is None:
        return ""
    if isinstance(x, (np.generic,)):
        x = np.asarray(x).item()
    if isinstance(x, float):
        return x if math.isfinite(x) else ""
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
    return out.astype(object)

def with_retries(fn, *args, retries: int = 3, delay: float = 0.8, **kwargs):
    last_ex = None
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except (TransportError, gspread.exceptions.APIError, ConnectionError) as e:
            last_ex = e
            time.sleep(delay * (1.5 ** i))
        except Exception:
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
            row = {k: json_safe(v) for k, v in row.items()}
            return row, None
    return None, "Contract not found"

def safe_ws_update(ws, range_name: str, values):
    safe_values = [[json_safe(v) for v in row] for row in values]
    return with_retries(ws.update, range_name=range_name, values=safe_values, retries=3, delay=0.7)

# ========= Theming =========
def theme_puts_table(ws, start_row: int, start_col: int, n_rows: int, n_cols: int, title_text: str):
    """
    Apply a clean visual theme to the puts section:
    - Merge & style the title row (start_row).
    - Bold header row (start_row+1).
    - Alternating banding for data rows (start_row+2 ..).
    - Number formats for numeric columns.
    - Freeze rows to keep the section header visible.
    """
    if not _HAS_FMT or n_cols <= 0:
        return

    # A1 ranges
    start_col_a1 = col_to_a1(start_col)
    end_col_a1   = col_to_a1(start_col + n_cols - 1)

    # 1) Title: merge across all columns and style
    try:
        ws.merge_cells(start_row, start_col, start_row, start_col + n_cols - 1)
    except Exception:
        pass

    try:
        format_cell_range(ws, f"{start_col_a1}{start_row}",
                          CellFormat(
                              backgroundColor=Color(0.10, 0.14, 0.25),
                              textFormat=TextFormat(bold=True, fontSize=12, foregroundColor=Color(1,1,1))
                          ))
    except Exception:
        pass

    # 2) Header row (DF header is start_row+1)
    header_row = start_row + 1
    try:
        format_cell_range(ws, f"{start_col_a1}{header_row}:{end_col_a1}{header_row}",
                          CellFormat(
                              backgroundColor=Color(0.18, 0.24, 0.38),
                              textFormat=TextFormat(bold=True, fontSize=10, foregroundColor=Color(1,1,1))
                          ))
    except Exception:
        pass

    # 3) Alternating banding for data area (if there are data rows)
    data_start = start_row + 2
    data_end   = start_row + n_rows  # includes header row + data rows
    if data_end >= data_start:
        try:
            add_banding(
                ws,
                GridRange(
                    worksheet=ws,
                    start_row_index=data_start - 1,  # 0-based
                    end_row_index=data_end,          # exclusive
                    start_column_index=start_col - 1,
                    end_column_index=start_col - 1 + n_cols
                ),
                theme=BandingTheme.BLUE
            )
        except Exception:
            pass

    # 4) Number formats: set reasonable defaults
    # Columns (by name): contractSymbol, strike, last, bid, ask, mid, openInterest, impliedVol, volume, inTheMoney
    # strike/last/bid/ask/mid -> 2 decimals; openInterest/volume -> integer; impliedVol -> percent
    # Build safe ranges by columns:
    name_to_idx = {
        "strike": 2, "last": 3, "bid": 4, "ask": 5, "mid": 6,
        "openInterest": 7, "impliedVol": 8, "volume": 9
    }
    # Convert DF-relative column indices to sheet columns
    def col_range(col_idx: int) -> str:
        abs_col = start_col + col_idx - 1
        a1 = col_to_a1(abs_col)
        return f"{a1}{data_start}:{a1}{data_end}"

    try:
        # 2-decimal money-like columns
        money_fmt = CellFormat(numberFormat=NumberFormat(type="NUMBER", pattern="#,##0.00"))
        for colname in ["strike","last","bid","ask","mid"]:
            if colname in name_to_idx:
                format_cell_range(ws, col_range(name_to_idx[colname]), money_fmt)
        # Integer columns
        int_fmt = CellFormat(numberFormat=NumberFormat(type="NUMBER", pattern="#,##0"))
        for colname in ["openInterest","volume"]:
            if colname in name_to_idx:
                format_cell_range(ws, col_range(name_to_idx[colname]), int_fmt)
        # Percent for IV
        pct_fmt = CellFormat(numberFormat=NumberFormat(type="PERCENT", pattern="0.00%"))
        if "impliedVol" in name_to_idx:
            format_cell_range(ws, col_range(name_to_idx["impliedVol"]), pct_fmt)
    except Exception:
        pass

    # 5) Freeze rows up to the table header so the section stays visible
    try:
        set_frozen(ws, rows=header_row)
    except Exception:
        pass

# ========= Main =========
def main():
    sheet_id = _sheet_id(SHEET_URL)
    sh = with_retries(gc.open_by_key, sheet_id, retries=3, delay=0.7)
    ws = with_retries(sh.worksheet, WORKSHEET_NAME, retries=3, delay=0.7)

    # A1 timestamp (America/Indiana/Indianapolis), 12-hour clock
    now_local = datetime.now(ZoneInfo("America/Indiana/Indianapolis"))
    # Example: "Updated: 7:45 PM — September 19, 2025 (ET)"
    ts_str = now_local.strftime("Updated: %I:%M %p — %B %d, %Y (ET)").lstrip("0")
    safe_ws_update(ws, "A1", [[ts_str]])

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

            # Section title
            safe_ws_update(ws, OUT_PUTS_HEADER_CELL, [[f"Puts @ {nearest}"]])

            # Write DF below title row
            title_row, title_col = a1_to_rowcol(OUT_PUTS_HEADER_CELL)
            data_row = title_row + 1
            data_col = title_col

            puts_df = df_json_safe(puts_df)
            with_retries(set_with_dataframe, ws, puts_df, row=data_row, col=data_col, retries=2, delay=0.6)

            # Apply theme over title + header + data
            n_rows = 1 + len(puts_df)  # header row + data rows
            n_cols = puts_df.shape[1] if not puts_df.empty else 10
            theme_puts_table(ws, start_row=title_row, start_col=title_col, n_rows=n_rows, n_cols=n_cols, title_text=f"Puts @ {nearest}")
        else:
            safe_ws_update(ws, OUT_EXPS_START_CELL, [["No expirations available"]])
    else:
        safe_ws_update(ws, OUT_EXPS_HEADER_CELL, [["(Put Ticker in A2)"]])

    print("Done (puts only).")

if __name__ == "__main__":
    main()
