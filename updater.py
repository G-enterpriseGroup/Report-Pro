# updater.py
import os, re, json, math, time
from typing import Any, Dict, Optional, Tuple, List
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
import gspread
from gspread_dataframe import set_with_dataframe
from google.oauth2.service_account import Credentials
from google.auth.exceptions import TransportError

try:
    from gspread_formatting import (
        CellFormat, Color, TextFormat, NumberFormat,
        format_cell_range, set_frozen, add_banding, BandingTheme, GridRange
    )
    _HAS_FMT = True
except Exception:
    _HAS_FMT = False

SHEET_URL       = os.environ["SHEET_URL"].strip()
SHEET_URL2      = os.environ.get("SHEET_URL2", "").strip()
WORKSHEET_NAME  = os.environ.get("WORKSHEET_NAME", "Copy").strip()

CELL_TICKER = "A2"
CELL_OCC    = "A3"

OUT_OCC_HEADER_CELL      = "A4"
OUT_OCC_VALUES_CELL      = "A5"
OUT_EXPS_HEADER_CELL     = "A7"
OUT_EXPS_START_CELL      = "A8"

OUT_SUMMARY_HEADER_CELL  = "C4"
OUT_SUMMARY_VALUES_CELL  = "C5"

OUT_ALL_HEADER_CELL      = "C7"   # main master table starts here; rows 1-2 are untouched

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
creds_info = json.loads(os.environ["GOOGLE_CREDENTIALS"])
creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
gc = gspread.authorize(creds)

def _sheet_id(url_or_id: str) -> str:
    s = url_or_id.strip().strip('"').strip("'")
    m = re.search(r"/spreadsheets/d/([A-Za-z0-9-_]+)", s)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9-_]{20,}", s):
        return s
    raise ValueError(f"Could not parse Sheet ID from value: {s[:80]}")

def a1_to_rowcol(a1: str) -> Tuple[int, int]:
    col_str = ''.join(filter(str.isalpha, a1))
    row_str = ''.join(filter(str.isdigit, a1))
    row = int(row_str)
    col = 0
    for i, c in enumerate(reversed(col_str.upper())):
        col += (ord(c) - 64) * (26 ** i)
    return row, col

def col_to_a1(col_index: int) -> str:
    s = ""
    n = col_index
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s

def json_safe(x: Any) -> Any:
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

def safe_ws_update(ws, range_name: str, values):
    safe_values = [[json_safe(v) for v in row] for row in values]
    return with_retries(ws.update, range_name=range_name, values=safe_values, retries=3, delay=0.7)

def clear_range(ws, start_a1: str, end_col: str = "Z", end_row: int = 5000):
    with_retries(ws.batch_clear, [f"{start_a1}:{end_col}{end_row}"], retries=2, delay=0.6)

def parse_occ(contract: str) -> Optional[Dict[str, Any]]:
    m = re.match(r"^([A-Za-z]+)(\d{2})(\d{2})(\d{2})([CP])(\d{8})$", (contract or "").strip())
    if not m:
        return None
    und, yy, mm, dd, cp, strike_code = m.groups()
    year = 2000 + int(yy)
    month = int(mm)
    day = int(dd)
    expiry_iso = f"{year:04d}-{month:02d}-{day:02d}"
    strike = int(strike_code) / 1000.0
    return {"underlying": und.upper(), "type": cp, "strike": strike, "expiry_iso": expiry_iso}

def _fmt_mid(bid, ask):
    try:
        if pd.notna(bid) and pd.notna(ask):
            b = float(bid)
            a = float(ask)
            if b > 0 and a > 0:
                return round((b + a) / 2, 4)
    except Exception:
        pass
    return ""

def normalize_option_side(df: pd.DataFrame, expiry: str, opt_type: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "expiry","type","contractSymbol","strike","last","bid","ask","mid",
            "openInterest","impliedVol","volume","inTheMoney"
        ])

    out = df.copy()
    required = [
        "contractSymbol","strike","lastPrice","bid","ask",
        "openInterest","impliedVolatility","volume","inTheMoney"
    ]
    for col in required:
        if col not in out.columns:
            out[col] = pd.NA

    out["expiry"] = expiry
    out["type"] = opt_type
    out["mid"] = out.apply(lambda r: _fmt_mid(r.get("bid"), r.get("ask")), axis=1)

    out = out.rename(columns={
        "lastPrice": "last",
        "impliedVolatility": "impliedVol",
    })[[
        "expiry","type","contractSymbol","strike","last","bid","ask","mid",
        "openInterest","impliedVol","volume","inTheMoney"
    ]]

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
                "contractSymbol": r.get("contractSymbol", ""),
                "underlying": meta["underlying"],
                "expiry": exp,
                "type": ("PUT" if meta["type"] == "P" else "CALL"),
                "strike": json_safe(r.get("strike", meta["strike"])),
                "last": json_safe(r.get("lastPrice", "")),
                "bid": json_safe(r.get("bid", "")),
                "ask": json_safe(r.get("ask", "")),
                "mid": json_safe(_fmt_mid(r.get("bid"), r.get("ask"))),
                "oi": json_safe(r.get("openInterest", "")),
                "iv": json_safe(r.get("impliedVolatility", "")),
                "volume": json_safe(r.get("volume", "")),
                "itm": json_safe(r.get("inTheMoney", "")),
                "currency": "USD",
            }
            row = {k: json_safe(v) for k, v in row.items()}
            return row, None

    return None, "Contract not found"

def fetch_all_contracts_for_ticker(ticker: str) -> Tuple[List[str], pd.DataFrame]:
    t = yf.Ticker(ticker)
    expiries = list(getattr(t, "options", []) or [])
    if not expiries:
        return [], pd.DataFrame(columns=[
            "expiry","type","contractSymbol","strike","last","bid","ask","mid",
            "openInterest","impliedVol","volume","inTheMoney"
        ])

    all_parts = []

    for exp in sorted(expiries):
        try:
            chain = with_retries(t.option_chain, exp, retries=2, delay=0.6)
        except Exception:
            continue

        calls_df = normalize_option_side(chain.calls, exp, "CALL")
        puts_df  = normalize_option_side(chain.puts,  exp, "PUT")

        if not calls_df.empty:
            all_parts.append(calls_df)
        if not puts_df.empty:
            all_parts.append(puts_df)

    if not all_parts:
        return expiries, pd.DataFrame(columns=[
            "expiry","type","contractSymbol","strike","last","bid","ask","mid",
            "openInterest","impliedVol","volume","inTheMoney"
        ])

    master = pd.concat(all_parts, ignore_index=True)
    master["type_sort"] = master["type"].map({"CALL": 0, "PUT": 1}).fillna(9)
    master["strike_num"] = pd.to_numeric(master["strike"], errors="coerce")

    master = master.sort_values(
        by=["expiry", "type_sort", "strike_num", "contractSymbol"],
        ascending=[True, True, True, True]
    ).drop(columns=["type_sort", "strike_num"], errors="ignore")

    return expiries, df_json_safe(master)

def theme_table(ws, start_row: int, start_col: int, n_rows: int, n_cols: int, title_text: str):
    if not _HAS_FMT or n_cols <= 0:
        return

    start_col_a1 = col_to_a1(start_col)
    end_col_a1   = col_to_a1(start_col + n_cols - 1)

    try:
        ws.merge_cells(start_row, start_col, start_row, start_col + n_cols - 1)
    except Exception:
        pass

    try:
        safe_ws_update(ws, f"{start_col_a1}{start_row}", [[title_text]])
    except Exception:
        pass

    try:
        format_cell_range(
            ws,
            f"{start_col_a1}{start_row}",
            CellFormat(
                backgroundColor=Color(0.10, 0.14, 0.25),
                textFormat=TextFormat(bold=True, fontSize=12, foregroundColor=Color(1, 1, 1))
            )
        )
    except Exception:
        pass

    header_row = start_row + 1
    try:
        format_cell_range(
            ws,
            f"{start_col_a1}{header_row}:{end_col_a1}{header_row}",
            CellFormat(
                backgroundColor=Color(0.18, 0.24, 0.38),
                textFormat=TextFormat(bold=True, fontSize=10, foregroundColor=Color(1, 1, 1))
            )
        )
    except Exception:
        pass

    data_start = start_row + 2
    data_end   = start_row + n_rows

    if data_end >= data_start:
        try:
            add_banding(
                ws,
                GridRange(
                    worksheet=ws,
                    start_row_index=data_start - 1,
                    end_row_index=data_end,
                    start_column_index=start_col - 1,
                    end_column_index=start_col - 1 + n_cols
                ),
                theme=BandingTheme.BLUE
            )
        except Exception:
            pass

    numeric_cols = {
        "strike": 4,
        "last": 5,
        "bid": 6,
        "ask": 7,
        "mid": 8,
        "openInterest": 9,
        "impliedVol": 10,
        "volume": 11
    }

    def col_range(col_idx: int) -> str:
        abs_col = start_col + col_idx - 1
        a1 = col_to_a1(abs_col)
        return f"{a1}{data_start}:{a1}{data_end}"

    try:
        money_fmt = CellFormat(numberFormat=NumberFormat(type="NUMBER", pattern="#,##0.00"))
        for colname in ["strike", "last", "bid", "ask", "mid"]:
            if colname in numeric_cols:
                format_cell_range(ws, col_range(numeric_cols[colname]), money_fmt)

        int_fmt = CellFormat(numberFormat=NumberFormat(type="NUMBER", pattern="#,##0"))
        for colname in ["openInterest", "volume"]:
            if colname in numeric_cols:
                format_cell_range(ws, col_range(numeric_cols[colname]), int_fmt)

        pct_fmt = CellFormat(numberFormat=NumberFormat(type="PERCENT", pattern="0.00%"))
        if "impliedVol" in numeric_cols:
            format_cell_range(ws, col_range(numeric_cols["impliedVol"]), pct_fmt)
    except Exception:
        pass

    try:
        set_frozen(ws, rows=2)
    except Exception:
        pass

def write_master_table(ws, header_cell: str, title_text: str, df: pd.DataFrame):
    title_row, title_col = a1_to_rowcol(header_cell)
    df = df_json_safe(df)

    if df.empty:
        safe_ws_update(ws, header_cell, [[title_text]])
        safe_ws_update(ws, f"{col_to_a1(title_col)}{title_row + 1}", [["No data available"]])
        return

    safe_ws_update(ws, header_cell, [[title_text]])
    with_retries(set_with_dataframe, ws, df, row=title_row + 1, col=title_col, retries=2, delay=0.6)

    n_rows = 1 + len(df)
    n_cols = df.shape[1]
    theme_table(ws, start_row=title_row, start_col=title_col, n_rows=n_rows, n_cols=n_cols, title_text=title_text)

def run_for_sheet(sheet_url: str, label: str):
    print(f"--- START {label} ---")
    print(f"{label} URL present? {'YES' if sheet_url else 'NO'}")

    sheet_id = _sheet_id(sheet_url)
    print(f"{label} sheet_id: {sheet_id}")

    sh = with_retries(gc.open_by_key, sheet_id, retries=3, delay=0.7)
    print(f"{label} spreadsheet title: {sh.title}")

    print(f"{label} available tabs: {[ws.title for ws in sh.worksheets()]}")
    ws = with_retries(sh.worksheet, WORKSHEET_NAME, retries=3, delay=0.7)
    print(f"{label} using worksheet: {ws.title}")

    now_local = datetime.now(ZoneInfo("America/Indiana/Indianapolis"))
    ts_str = now_local.strftime("Updated: %I:%M %p — %B %d, %Y (ET)").lstrip("0")

    # keep row 1 and 2 safe except A1 timestamp
    safe_ws_update(ws, "A1", [[ts_str]])

    ticker = (ws.acell(CELL_TICKER).value or "").strip().upper()
    occ    = (ws.acell(CELL_OCC).value or "").strip().upper()
    print(f"{label} ticker in A2: {ticker}")
    print(f"{label} occ in A3: {occ}")

    # clear only output area below row 3
    clear_range(ws, "A4", end_col="Z", end_row=5000)

    headers = [[
        "ContractSymbol","Underlying","Expiry","Type","Strike","Last",
        "Bid","Ask","Mid","OpenInterest","ImpliedVol","Volume","InTheMoney","Currency"
    ]]
    safe_ws_update(ws, OUT_OCC_HEADER_CELL, headers)

    if occ:
        row, err = lookup_occ_with_yf(occ)
        if row:
            safe_ws_update(ws, OUT_OCC_VALUES_CELL, [[
                row["contractSymbol"], row["underlying"], row["expiry"], row["type"], row["strike"],
                row["last"], row["bid"], row["ask"], row["mid"], row["oi"], row["iv"],
                row["volume"], row["itm"], row["currency"]
            ]])
        else:
            safe_ws_update(ws, OUT_OCC_VALUES_CELL, [[err]])
    else:
        safe_ws_update(ws, OUT_OCC_VALUES_CELL, [["(Put OCC in A3)"]])

    safe_ws_update(ws, OUT_SUMMARY_HEADER_CELL, [["Ticker", "Expirations", "Total Contracts", "Calls", "Puts"]])

    if ticker:
        expiries, master_df = fetch_all_contracts_for_ticker(ticker)

        safe_ws_update(ws, OUT_EXPS_HEADER_CELL, [["Expirations (ISO)"]])
        if expiries:
            safe_ws_update(ws, OUT_EXPS_START_CELL, [[d] for d in expiries])

        total_contracts = len(master_df)
        call_count = int((master_df["type"] == "CALL").sum()) if not master_df.empty else 0
        put_count  = int((master_df["type"] == "PUT").sum()) if not master_df.empty else 0

        safe_ws_update(ws, OUT_SUMMARY_VALUES_CELL, [[
            ticker,
            len(expiries),
            total_contracts,
            call_count,
            put_count
        ]])

        write_master_table(ws, OUT_ALL_HEADER_CELL, f"All Option Contracts for {ticker}", master_df)
        print(f"{label} wrote all contracts for {ticker}: {total_contracts} rows across {len(expiries)} expirations")
    else:
        safe_ws_update(ws, OUT_EXPS_HEADER_CELL, [["(Put Ticker in A2)"]])
        safe_ws_update(ws, OUT_SUMMARY_VALUES_CELL, [["", "", "", "", ""]])
        safe_ws_update(ws, OUT_ALL_HEADER_CELL, [["(No ticker in A2)"]])
        print(f"{label} no ticker in A2")

    print(f"--- END {label} ---")

def main():
    print("service account:", creds_info.get("client_email"))
    print("WORKSHEET_NAME:", WORKSHEET_NAME)
    print("SHEET_URL exists?", "YES" if SHEET_URL else "NO")
    print("SHEET_URL2 exists?", "YES" if SHEET_URL2 else "NO")

    run_for_sheet(SHEET_URL, "SHEET 1")

    if SHEET_URL2:
        run_for_sheet(SHEET_URL2, "SHEET 2")
    else:
        print("SHEET_URL2 is blank, skipping sheet 2")

    print("Done.")

if __name__ == "__main__":
    main()
