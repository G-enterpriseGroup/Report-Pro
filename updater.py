import os, re, json
import pandas as pd
import yfinance as yf
import gspread
from gspread_dataframe import set_with_dataframe
from google.oauth2.service_account import Credentials

# --------- Read required secrets (from GitHub Actions) ---------
SHEET_URL      = os.environ["SHEET_URL"].strip()                 # set in repo secrets
WORKSHEET_NAME = os.environ.get("WORKSHEET_NAME", "Copy").strip()# set in repo secrets

# --------- A1 locations ---------
CELL_TICKER = "A2"
CELL_OCC    = "A3"

OUT_OCC_HEADER_CELL   = "A4"
OUT_OCC_VALUES_CELL   = "A5"
OUT_EXPS_HEADER_CELL  = "A7"
OUT_EXPS_START_CELL   = "A8"
OUT_PUTS_HEADER_CELL  = "C7"

# --------- Google auth (Service Account JSON in GOOGLE_CREDENTIALS) ---------
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
creds_info = json.loads(os.environ["GOOGLE_CREDENTIALS"])
creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
gc = gspread.authorize(creds)

def _sheet_id(url_or_id: str) -> str:
    s = url_or_id.strip().strip('"').strip("'")
    m = re.search(r"/spreadsheets/d/([A-Za-z0-9-_]+)", s)
    if m: return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9-_]{20,}", s): return s  # raw ID
    raise ValueError(f"Could not parse Sheet ID from SHEET_URL='{s[:80]}'")

SHEET_ID = _sheet_id(SHEET_URL)
sh = gc.open_by_key(SHEET_ID)
ws = sh.worksheet(WORKSHEET_NAME)

# --------- Helpers ---------
def parse_occ(contract: str):
    m = re.match(r"^([A-Za-z]+)(\d{2})(\d{2})(\d{2})([CP])(\d{8})$", (contract or "").strip())
    if not m: return None
    und, yy, mm, dd, cp, strike_code = m.groups()
    year  = 2000 + int(yy); month = int(mm); day = int(dd)
    expiry_iso = f"{year:04d}-{month:02d}-{day:02d}"
    strike = int(strike_code) / 1000.0
    return {"underlying": und, "type": cp, "strike": strike, "expiry_iso": expiry_iso}

def _fmt_mid(bid, ask):
    try:
        if pd.notna(bid) and pd.notna(ask) and float(bid) > 0 and float(ask) > 0:
            return round((float(bid) + float(ask)) / 2, 4)
    except Exception:
        pass
    return ""

def normalize_puts(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    out = df.copy()
    for col in ["contractSymbol","strike","lastPrice","bid","ask","openInterest","impliedVolatility","volume","inTheMoney"]:
        if col not in out.columns: out[col] = pd.NA
    out["mid"] = out.apply(lambda r: _fmt_mid(r.get("bid"), r.get("ask")), axis=1)
    out = out.rename(columns={
        "lastPrice":"last",
        "openInterest":"openInterest",
        "impliedVolatility":"impliedVol"
    })[["contractSymbol","strike","last","bid","ask","mid","openInterest","impliedVol","volume","inTheMoney"]]
    return out

def a1_to_rowcol(a1: str):
    col_str = ''.join(filter(str.isalpha, a1))
    row_str = ''.join(filter(str.isdigit, a1))
    row = int(row_str); col = 0
    for i, c in enumerate(reversed(col_str.upper())):
        col += (ord(c) - 64) * (26 ** i)
    return row, col

def lookup_occ_with_yf(occ: str):
    meta = parse_occ(occ)
    if not meta: return None, "Invalid OCC contract"
    t = yf.Ticker(meta["underlying"])
    expiries = list(t.options or [])
    if not expiries: return None, f"No expirations for {meta['underlying']}"
    search_order = [meta["expiry_iso"]] + [e for e in expiries if e != meta["expiry_iso"]] if meta["expiry_iso"] in expiries else expiries
    for exp in search_order:
        try:
            chain = t.option_chain(exp)
        except Exception:
            continue
        pool = chain.puts if meta["type"] == "P" else chain.calls
        exact = pool[pool["contractSymbol"] == occ]
        if not exact.empty:
            r = exact.iloc[0].to_dict()
            return {
                "contractSymbol": r.get("contractSymbol",""),
                "underlying": meta["underlying"],
                "expiry": exp,
                "type": ("PUT" if meta["type"]=="P" else "CALL"),
                "strike": r.get("strike", meta["strike"]),
                "last": r.get("lastPrice",""),
                "bid": r.get("bid",""),
                "ask": r.get("ask",""),
                "mid": _fmt_mid(r.get("bid"), r.get("ask")),
                "oi": r.get("openInterest",""),
                "iv": r.get("impliedVolatility",""),
                "volume": r.get("volume",""),
                "itm": r.get("inTheMoney",""),
                "currency": "USD",
            }, None
    return None, "Contract not found"

# --------- Read inputs from sheet (A2/A3) ---------
ticker = (ws.acell(CELL_TICKER).value or "").strip().upper()
occ    = (ws.acell(CELL_OCC).value or "").strip().upper()

# --------- OCC snapshot row ---------
headers = ["ContractSymbol","Underlying","Expiry","Type","Strike","Last","Bid","Ask","Mid","OpenInterest","ImpliedVol","Volume","InTheMoney","Currency"]
ws.update(OUT_OCC_HEADER_CELL, [headers])
if occ:
    row, err = lookup_occ_with_yf(occ)
    if row:
        ws.update(OUT_OCC_VALUES_CELL, [[
            row["contractSymbol"], row["underlying"], row["expiry"], row["type"], row["strike"],
            row["last"], row["bid"], row["ask"], row["mid"], row["oi"], row["iv"], row["volume"], row["itm"], row["currency"]
        ]])
    else:
        ws.update(OUT_OCC_VALUES_CELL, [[err]])
else:
    ws.update(OUT_OCC_VALUES_CELL, [["(Put OCC in A3)"]])

# --------- Expirations + nearest puts for A2 ticker ---------
if ticker:
    t = yf.Ticker(ticker)
    expiries = list(t.options or [])
    ws.update(OUT_EXPS_HEADER_CELL, [["Expirations (ISO)"]])
    if expiries:
        ws.update(OUT_EXPS_START_CELL, [[d] for d in expiries])
        nearest = sorted(expiries)[0]
        chain = t.option_chain(nearest)
        puts_df = normalize_puts(chain.puts)
        ws.update(OUT_PUTS_HEADER_CELL, [[f"Puts @ {nearest}"]])
        r, c = a1_to_rowcol(OUT_PUTS_HEADER_CELL)
        set_with_dataframe(ws, puts_df, row=r+1, col=c)
    else:
        ws.update(OUT_EXPS_START_CELL, [["No expirations available"]])
else:
    ws.update(OUT_EXPS_HEADER_CELL, [["(Put Ticker in A2)"]])

print("Done (puts only).")
