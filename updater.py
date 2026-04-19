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
        format_cell_range, set_frozen,
        Border, Borders
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

OUT_ALL_HEADER_CELL      = "C7"   # grouped master section begins here

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

def fetch_all_contracts_grouped(ticker: str) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
    t = yf.Ticker(ticker)
    expiries = list(getattr(t, "options", []) or [])
    grouped = {}

    for exp in sorted(expiries):
        try:
            chain = with_retries(t.option_chain, exp, retries=2, delay=0.6)
        except Exception:
            continue

        calls_df = normalize_option_side(chain.calls, exp, "CALL")
        puts_df  = normalize_option_side(chain.puts,  exp, "PUT")

        parts = []
        if not calls_df.empty:
            calls_df["strike_num"] = pd.to_numeric(calls_df["strike"], errors="coerce")
            calls_df = calls_df.sort_values(["strike_num", "contractSymbol"]).drop(columns=["strike_num"], errors="ignore")
            parts.append(calls_df)

        if not puts_df.empty:
            puts_df["strike_num"] = pd.to_numeric(puts_df["strike"], errors="coerce")
            puts_df = puts_df.sort_values(["strike_num", "contractSymbol"]).drop(columns=["strike_num"], errors="ignore")
            parts.append(puts_df)

        if parts:
            grouped[exp] = pd.concat(parts, ignore_index=True)

    return expiries, grouped

def format_money_int_pct(ws, start_row: int, start_col: int, num_data_rows: int):
    if not _HAS_FMT or num_data_rows <= 0:
        return

    data_start = start_row
    data_end = start_row + num_data_rows - 1

    def rng(col_offset: int) -> str:
        c = col_to_a1(start_col + col_offset)
        return f"{c}{data_start}:{c}{data_end}"

    try:
        money_fmt = CellFormat(numberFormat=NumberFormat(type="NUMBER", pattern="#,##0.00"))
        for offset in [3, 4, 5, 6, 7]:  # strike,last,bid,ask,mid
            format_cell_range(ws, rng(offset), money_fmt)

        int_fmt = CellFormat(numberFormat=NumberFormat(type="NUMBER", pattern="#,##0"))
        for offset in [8, 10]:  # openInterest, volume
            format_cell_range(ws, rng(offset), int_fmt)

        pct_fmt = CellFormat(numberFormat=NumberFormat(type="PERCENT", pattern="0.00%"))
        format_cell_range(ws, rng(9), pct_fmt)  # impliedVol
    except Exception:
        pass

def style_range(ws, a1_range: str, fmt):
    if not _HAS_FMT:
        return
    try:
        format_cell_range(ws, a1_range, fmt)
    except Exception:
        pass

def write_grouped_master_table(ws, header_cell: str, ticker: str, grouped_data: Dict[str, pd.DataFrame]):
    title_row, title_col = a1_to_rowcol(header_cell)
    start_col_a1 = col_to_a1(title_col)
    end_col_a1 = col_to_a1(title_col + 11)  # 12 columns

    # Title row
    safe_ws_update(ws, header_cell, [[f"All Option Contracts for {ticker}"]])

    if _HAS_FMT:
        try:
            ws.merge_cells(title_row, title_col, title_row, title_col + 11)
        except Exception:
            pass

        style_range(
            ws,
            f"{start_col_a1}{title_row}:{end_col_a1}{title_row}",
            CellFormat(
                backgroundColor=Color(0.10, 0.14, 0.25),
                textFormat=TextFormat(bold=True, fontSize=12, foregroundColor=Color(1, 1, 1))
            )
        )

    current_row = title_row + 1

    all_written_rows = 0

    expiry_banner_fmt = CellFormat(
        backgroundColor=Color(0.30, 0.30, 0.30),
        textFormat=TextFormat(bold=True, foregroundColor=Color(1, 1, 1))
    )
    header_fmt = CellFormat(
        backgroundColor=Color(0.18, 0.24, 0.38),
        textFormat=TextFormat(bold=True, foregroundColor=Color(1, 1, 1))
    )
    call_fmt = CellFormat(
        backgroundColor=Color(0.88, 0.95, 1.00),
        textFormat=TextFormat(foregroundColor=Color(0.05, 0.20, 0.40))
    )
    put_fmt = CellFormat(
        backgroundColor=Color(1.00, 0.91, 0.91),
        textFormat=TextFormat(foregroundColor=Color(0.40, 0.05, 0.05))
    )
    thick_bottom_fmt = CellFormat(
        borders=Borders(
            bottom=Border(style="SOLID_THICK", color=Color(0.15, 0.15, 0.15))
        )
    )

    headers = [[
        "expiry","type","contractSymbol","strike","last","bid","ask","mid",
        "openInterest","impliedVol","volume","inTheMoney"
    ]]

    for exp in sorted(grouped_data.keys()):
        df = df_json_safe(grouped_data[exp])
        if df.empty:
            continue

        # Expiry banner
        safe_ws_update(ws, f"{start_col_a1}{current_row}", [[f"EXPIRATION: {exp}"]])
        if _HAS_FMT:
            try:
                ws.merge_cells(current_row, title_col, current_row, title_col + 11)
            except Exception:
                pass
            style_range(ws, f"{start_col_a1}{current_row}:{end_col_a1}{current_row}", expiry_banner_fmt)
        current_row += 1

        # Column headers
        safe_ws_update(ws, f"{start_col_a1}{current_row}", headers)
        style_range(ws, f"{start_col_a1}{current_row}:{end_col_a1}{current_row}", header_fmt)
        current_row += 1

        # Data rows
        with_retries(set_with_dataframe, ws, df, row=current_row, col=title_col, include_column_header=False, retries=2, delay=0.6)

        num_rows = len(df)
        if num_rows > 0:
            calls_mask = (df["type"] == "CALL").tolist()
            puts_mask  = (df["type"] == "PUT").tolist()

            call_start = None
            put_start = None

            # Since data is calls first then puts
            for i, val in enumerate(calls_mask):
                if val:
                    call_start = current_row + i
                    break

            for i, val in enumerate(puts_mask):
                if val:
                    put_start = current_row + i
                    break

            if call_start is not None:
                call_count = int((df["type"] == "CALL").sum())
                style_range(ws, f"{start_col_a1}{call_start}:{end_col_a1}{call_start + call_count - 1}", call_fmt)

            if put_start is not None:
                put_count = int((df["type"] == "PUT").sum())
                style_range(ws, f"{start_col_a1}{put_start}:{end_col_a1}{put_start + put_count - 1}", put_fmt)

            format_money_int_pct(ws, current_row, title_col, num_rows)

            # Thick bottom border at end of expiry block
            last_data_row = current_row + num_rows - 1
            style_range(ws, f"{start_col_a1}{last_data_row}:{end_col_a1}{last_data_row}", thick_bottom_fmt)

        current_row += num_rows

        # Blank spacer row
        current_row += 1
        all_written_rows += num_rows

    try:
        set_frozen(ws, rows=2)
    except Exception:
        pass

    return all_written_rows

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

    # Keep rows 1 and 2 safe except A1 timestamp
    safe_ws_update(ws, "A1", [[ts_str]])

    ticker = (ws.acell(CELL_TICKER).value or "").strip().upper()
    occ    = (ws.acell(CELL_OCC).value or "").strip().upper()
    print(f"{label} ticker in A2: {ticker}")
    print(f"{label} occ in A3: {occ}")

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
        expiries, grouped_data = fetch_all_contracts_grouped(ticker)

        safe_ws_update(ws, OUT_EXPS_HEADER_CELL, [["Expirations (ISO)"]])
        if expiries:
            safe_ws_update(ws, OUT_EXPS_START_CELL, [[d] for d in expiries])

        total_contracts = 0
        call_count = 0
        put_count = 0

        for exp, df in grouped_data.items():
            total_contracts += len(df)
            if not df.empty:
                call_count += int((df["type"] == "CALL").sum())
                put_count  += int((df["type"] == "PUT").sum())

        safe_ws_update(ws, OUT_SUMMARY_VALUES_CELL, [[
            ticker,
            len(expiries),
            total_contracts,
            call_count,
            put_count
        ]])

        written = write_grouped_master_table(ws, OUT_ALL_HEADER_CELL, ticker, grouped_data)
        print(f"{label} wrote grouped contracts for {ticker}: {written} rows across {len(expiries)} expirations")
        print(f"{label} formatting enabled? {_HAS_FMT}")
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
    print("gspread_formatting enabled?", _HAS_FMT)

    run_for_sheet(SHEET_URL, "SHEET 1")

    if SHEET_URL2:
        run_for_sheet(SHEET_URL2, "SHEET 2")
    else:
        print("SHEET_URL2 is blank, skipping sheet 2")

    print("Done.")

if __name__ == "__main__":
    main()
