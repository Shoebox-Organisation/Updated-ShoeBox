import os
from pathlib import Path
from datetime import date, timedelta, datetime
import base64
import re
import unicodedata
import calendar

import streamlit as st
import pandas as pd
import plotly.express as px

from dotenv import load_dotenv
from fpdf import FPDF

# Networking
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import certifi
import ssl

# --- PAGE CONFIG (must be first Streamlit call) ---
st.set_page_config(page_title="Shoebox Dashboard", layout="wide")

# Your Checkfront account's timezone
TENANT_TZ = "Europe/London"

# ---- Secrets / env (portable: Streamlit Cloud or local .env) ----
load_dotenv()  # harmless in cloud, loads local .env in dev

def _get_secret(name: str, default: str | None = None) -> str | None:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)

API_KEY = _get_secret("API_KEY")
API_TOKEN = _get_secret("API_TOKEN")

# Base URL: default to your tenant URL; override via secrets if needed
API_BASE = _get_secret(
    "CHECKFRONT_API_BASE",
    "https://theshoebox.checkfront.co.uk/api/3.0/booking"
)

# Optional: temporary escape hatch in cloud (ONLY for short-term debugging)
ALLOW_INSECURE = str(_get_secret("ALLOW_INSECURE_SSL", "false")).lower() == "true"

# (Only needed if you ever switch back to a global host that requires it)
CHECKFRONT_ACCOUNT = _get_secret("CHECKFRONT_ACCOUNT", "theshoebox")  # your subdomain w/o TLD



# --- TLS trust setup (Windows-friendly) ---
USING_OS_TRUST = False
try:
    import truststore
    truststore.inject_into_ssl()  # Use system trust store (works great on Windows/macOS)
    USING_OS_TRUST = True
except Exception:
    # Fallback: use certifi bundle (Linux/cloud)
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

# --- VAT config ---
VAT_RATE = 0.20  # change here if needed


def ex_vat(amount: float | int | None, rate: float = VAT_RATE) -> float:
    """Fallback ex-VAT calculation when tax_total is unavailable."""
    try:
        return float(amount) / (1.0 + rate)
    except Exception:
        return 0.0

# --- Tour allow-list + normaliser ---
def _norm_title(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).strip().lower()
    s = re.sub(r"\s+", " ", s)      # collapse spaces
    s = s.replace("–", "-")         # en-dash → hyphen
    return s

# --- Helpers for multi-item summaries ---
SPLIT_RE = re.compile(r"\s*(?:,|/|&|\+| and )\s*")  # splits on commas, slashes, &, +, " and "

def _parts(summary: str) -> list[str]:
    s = _norm_title(summary)
    if not s:
        return []
    return [p for p in SPLIT_RE.split(s) if p]

def contains_allowed_tour(summary: str) -> bool:
    parts = _parts(summary)
    return any(p in TOURS_ALLOWLIST for p in parts)

def _matches_status(row) -> bool:
    if status_filter == "All":
        return True
    return str(row.get("status_name", "")) == status_filter

def _matches_search(row, s: str) -> bool:
    if not s:
        return True
    s = s.strip().lower()
    return (
        s in str(row.get("customer_name", "")).lower()
        or s in str(row.get("customer_email", "")).lower()
        or s in str(row.get("code", "")).lower()
        or s in str(row.get("booking_id", "")).lower()
    )

def _matches_category(row) -> bool:
    if category_filter == "All":
        return True
    return str(row.get("product_category", "")) == category_filter

def _passes_tour_allow(row) -> bool:
    # Only enforce when Tour category selected
    if category_filter != "Tour":
        return True
    return contains_allowed_tour(str(row.get("summary", "")))

def _matches_specific_product(row) -> bool:
    if specific_product == "All":
        return True
    target = _norm_title(specific_product)
    return target in _parts(str(row.get("summary", "")))

TOURS_ALLOWLIST_RAW = [
    "Great Yarmouth - Seafront Tour",
    "The TIPSY Tavern Trail Tour",
    "The Tavern Trail Tour",
    "The Norwich Knowledge Tour",
    "Magnificent Marble Hall",
    "City of Centuries Tour",
    "The Matriarchs, Mayors & Merchants Tour",
    "Norwich's Hidden Street Tour - Family fun!",
    "Norwich's Hidden Street Tour",
]
TOURS_ALLOWLIST = {_norm_title(x) for x in TOURS_ALLOWLIST_RAW}

# ---- HTTPS session (retries, timeouts, TLS verify) ----
@st.cache_resource
def get_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=5, connect=5, read=5, backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
    )
    s.mount("https://", HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20))
    s.mount("http://",  HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20))

    if not USING_OS_TRUST:
        s.verify = certifi.where()   # Linux/cloud path

    if ALLOW_INSECURE:
        s.verify = False  # ⚠️ temporary only; remove when SSL chain is fixed

    # inject default timeout if caller doesn't pass one
    s.request = _with_default_timeout(s.request, timeout=30)
    return s


def _with_default_timeout(request_func, timeout: int):
    """Wrap Session.request to inject a default timeout if none provided."""
    def wrapper(method, url, **kwargs):
        kwargs.setdefault("timeout", timeout)
        return request_func(method, url, **kwargs)
    return wrapper


SESSION = get_session()

# --- Auth header ---
def get_auth_header():
    token = f"{API_KEY}:{API_TOKEN}"
    b64 = base64.b64encode(token.encode()).decode()
    h = {
        "Authorization": f"Basic {b64}",
        "Accept": "application/json",
        "User-Agent": "ShoeboxDashboard/1.0",
    }
    # If we're using the global API host, tell it which account
    if "api.checkfront.com" in API_BASE:
        h["X-Checkfront-Account"] = CHECKFRONT_ACCOUNT  # e.g., "theshoebox"
    return h


# --- Fetch bookings (paginated) ---
def fetch_bookings(start_date, end_date, limit=250, include_items=False, max_pages=4000, filter_on="created"):
    API_BASE = os.getenv("CHECKFRONT_API_BASE", "https://api.checkfront.com/3.0/booking")
    headers = get_auth_header()
    page = 1
    seen, out = set(), []

    while page <= max_pages:
        params = [("limit", limit), ("page", page)]
        if filter_on == "created":
            params.append(("created_date", f">{start_date.isoformat()}"))
            params.append(("created_date", f"<{(end_date + timedelta(days=1)).isoformat()}"))
        else:
            params.append(("start_date", start_date.isoformat()))
            params.append(("end_date", end_date.isoformat()))
        params.append(("sort", "created_date"))
        params.append(("dir", "asc"))
        if include_items:
            params.append(("expand", "items"))

        # --- Make the request ---
        r = SESSION.get(API_BASE, headers=headers, params=params)

        # --- Custom error handling (instead of try/except) ---
        if not r.ok:
            msg = f"{r.status_code} {r.reason} — {r.url}"
            body = r.text
            st.error(f"API error: {msg}\n\n{body[:500]}")
            raise requests.HTTPError(msg, response=r)

        data = r.json()
        page_rows = list((data.get("booking/index") or {}).values())
        if not page_rows:
            break

        for b in page_rows:
            bid = b.get("booking_id")
            if bid and bid not in seen:
                seen.add(bid)
                out.append(b)

        if len(page_rows) < limit:
            break
        page += 1

    return {"booking/index": {i: b for i, b in enumerate(out)}}


def fetch_booking_details(booking_id: str | int):
    url = f"https://theshoebox.checkfront.co.uk/api/3.0/booking/{booking_id}"
    r = SESSION.get(url, headers=get_auth_header(), params={"expand": "items"}, timeout=15)

    if not r.ok:
        msg = f"{r.status_code} {r.reason} — {r.url}"
        body = r.text
        st.error(f"API error: {msg}\n\n{body[:500]}")
        raise requests.HTTPError(msg, response=r)

    return r.json()

# --- Cache API results ---
@st.cache_data(ttl=300)
def get_raw(start, end, include_items=False, filter_on="created"):
    return fetch_bookings(start, end, include_items=include_items, filter_on=filter_on)


# --- Categorisation helper ---
CATEGORY_ORDER = ["Tour", "Group", "Room Hire", "Voucher", "Merchandise", "Fee", "Other"]

def categorise_product(summary: str) -> str:
    ns = _norm_title(summary)
    if not ns:
        return "Other"

    # If *any* part of the summary is an allowed tour, classify as Tour
    if contains_allowed_tour(summary):
        return "Tour"

    # Otherwise fall back to keywords
    s = ns
    if re.search(r"\broom\b|meeting|hire", s): return "Room Hire"
    if "group" in s: return "Group"
    if "voucher" in s or "gift" in s: return "Voucher"
    if "guidebook" in s or "souvenir" in s or "merch" in s: return "Merchandise"
    if "fee" in s or "reschedul" in s or "cancell" in s or "admin" in s: return "Fee"
    return "Other"



@st.cache_data(ttl=300)
def prepare_df(raw):
    df = pd.DataFrame(list(raw.get("booking/index", {}).values()))
    if df.empty:
        return df

    # --- created_date: build from any plausible source, normalize to Europe/London
    def _to_local_ts(series_like):
        s_epoch = pd.to_datetime(series_like, unit="s", errors="coerce", utc=True)
        s_iso   = pd.to_datetime(series_like, errors="coerce", utc=True)
        s = s_epoch.fillna(s_iso)
        s = s.dt.tz_convert(TENANT_TZ).dt.tz_localize(None)
        return s

    created = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for col in ("created_date", "created", "created_at", "date_created", "timestamp_created"):
        if col in df.columns:
            s = _to_local_ts(df[col])
            created = created.fillna(s)
    df["created_date"] = created

    # numerics
    for col in ("total", "tax_total"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # labels / helpers
    df["status_name"] = df.get("status_name", "Unknown").fillna("Unknown")
    df["summary"] = df.get("summary", "").astype(str).str.strip()
    df["day"]  = df["created_date"].dt.day_name()
    df["hour"] = df["created_date"].dt.hour

    # --- category & ex-VAT ---
    df["product_category"] = df["summary"].apply(categorise_product)

    if "tax_total" in df.columns:
        # API provides explicit tax amount
        tax = pd.to_numeric(df["tax_total"], errors="coerce").fillna(0.0)
        df["total_ex_vat"] = (df["total"].fillna(0.0) - tax).clip(lower=0.0)

    elif "Taxes" in df.columns:
        # Excel exports provide 'Taxes' column
        tax = pd.to_numeric(df["Taxes"], errors="coerce").fillna(0.0)
        df["total_ex_vat"] = (df["total"].fillna(0.0) - tax).clip(lower=0.0)

    else:
        # Fallback: assume flat VAT rate
        df["total_ex_vat"] = df["total"].fillna(0.0).apply(ex_vat)

    # --- de-dupe
    if "booking_id" in df.columns:
        df = df.drop_duplicates(subset="booking_id", keep="last")
    else:
        df = df.drop_duplicates()

    return df

# --- Helper: extract event_date from items (robust) ---
def _extract_event_dt_from_items(items):
    if isinstance(items, dict): items = list(items.values())
    if not isinstance(items, list): return pd.NaT
    cands = []
    for it in items:
        if not isinstance(it, dict): continue
        for src in (it, it.get("date") if isinstance(it.get("date"), dict) else None):
            if not isinstance(src, dict): continue
            for key in ("start","start_date","date_start","from","event_date","date","datetime"):
                v = src.get(key)
                if v is None: continue
                dt = pd.to_datetime(v, unit="s", errors="coerce") if isinstance(v,(int,float)) else pd.to_datetime(v, errors="coerce")
                if pd.notna(dt): cands.append(dt)
        v = it.get("date_desc")
        if v:
            dt = pd.to_datetime(v, errors="coerce")
            if pd.notna(dt): cands.append(dt)
    return min(cands) if cands else pd.NaT

# --- PDF report function (now respects VAT toggle) ---
def create_detailed_pdf_summary(kpi_data, date_range, top_tour, top_day, recent_rows, logo_path=None, use_ex_vat=False):
    pdf = FPDF()
    pdf.add_page()
    if logo_path and Path(logo_path).exists():
        pdf.image(str(logo_path), x=10, y=8, w=33)
        pdf.set_xy(50, 10)
    else:
        pdf.set_y(15)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Shoebox Sales Summary Report", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Date Range: {date_range}", ln=True)
    pdf.ln(2)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Key Metrics", ln=True)
    pdf.set_font("Arial", "", 12)
    for label, value in kpi_data.items():
        pdf.cell(0, 8, f"{label}: {value}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Top Performer", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Best-Selling Tour: {top_tour}", ln=True)
    pdf.cell(0, 8, f"Most Popular Day: {top_day}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Recent Bookings", ln=True)
    pdf.set_font("Arial", "", 11)

    col_widths = [15, 50, 35, 20, 30]
    amount_hdr = "Amount (ex VAT)" if use_ex_vat else "Amount"
    headers = ["#", "Customer", amount_hdr, "Status", "Date"]
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 8, header, border=1)
    pdf.ln()

    for row in recent_rows:
        total_val = float(row.get("total", 0) or 0)
        if use_ex_vat:
            tax_val = row.get("tax_total", None)
            if tax_val is not None:
                try:
                    net = float(total_val) - float(tax_val or 0)
                except Exception:
                    net = ex_vat(total_val)
            else:
                net = ex_vat(total_val)
            amount_to_show = net
        else:
            amount_to_show = total_val

        pdf.cell(col_widths[0], 8, str(row.get("booking_id","")), border=1)
        pdf.cell(col_widths[1], 8, str(row.get("customer_name",""))[:24], border=1)
        pdf.cell(col_widths[2], 8, f"£{amount_to_show:.2f}", border=1)
        pdf.cell(col_widths[3], 8, str(row.get("status_name","")), border=1)
        dt = row.get("created_date")
        date_str = datetime.strftime(dt, "%Y-%m-%d") if isinstance(dt, datetime) else str(dt)[:10]
        pdf.cell(col_widths[4], 8, date_str, border=1)
        pdf.ln()

    return pdf.output(dest="S").encode("latin-1")

# --- Header ---
st.markdown("<h1 style='text-align: center;'>Shoebox Internal Operations Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- Sidebar filters ---
with st.sidebar.form("filters"):
    logo = Path(__file__).parent / "shoebox.png"
    if logo.exists():
        st.image(str(logo), width=180)

    today = date.today()
    start = st.date_input("Start Date", today - timedelta(days=30))
    end = st.date_input("End Date", today + timedelta(days=60))
    search = st.text_input("🔍 Search name, email or booking code").lower()

    # Build options from lightweight pull
    temp_raw = get_raw(start, end, include_items=False)
    temp_df = prepare_df(temp_raw)
    status_options = ["All"] + (sorted(temp_df["status_name"].dropna().unique()) if not temp_df.empty else [])
    status_filter = st.selectbox("Filter by Booking Status", status_options)

    date_basis = st.selectbox(
        "Date basis for KPIs & charts",
        ["Booking date (created)", "Event date"],
        index=0
    )

    # Product filters
    category_options = ["All"] + CATEGORY_ORDER
    category_filter = st.selectbox("Product category", category_options, index=0)

    if not temp_df.empty:
        if category_filter == "All":
            products_in_cat = sorted(temp_df["summary"].dropna().unique())
        elif category_filter == "Tour":
            present = temp_df["summary"].dropna().astype(str).unique().tolist()
            products_in_cat = sorted(p for p in present if _norm_title(p) in TOURS_ALLOWLIST)
        else:
            products_in_cat = sorted(
                temp_df.loc[temp_df["product_category"] == category_filter, "summary"].dropna().unique()
            )
    else:
        products_in_cat = []
    specific_product = st.selectbox("Specific product (within category)", ["All"] + products_in_cat, index=0)

    submitted = st.form_submit_button("Apply filters")

if not submitted and "data_ready" not in st.session_state:
    st.stop()
st.session_state.data_ready = True

AMOUNT_COL   = "total_ex_vat"
AMOUNT_LABEL = "Amount (ex VAT)"



def _apply_filters(dfx: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy so we don't mutate callers
    dfx = dfx.copy()

    # Ensure required derived columns exist
    if "summary" not in dfx.columns:
        dfx["summary"] = ""
    if "status_name" not in dfx.columns:
        dfx["status_name"] = "Unknown"
    if "product_category" not in dfx.columns:
        dfx["product_category"] = dfx["summary"].apply(categorise_product)
    if "total_ex_vat" not in dfx.columns:
        total_num = pd.to_numeric(dfx.get("total", 0.0), errors="coerce").fillna(0.0)
        if "tax_total" in dfx.columns:
            tax = pd.to_numeric(dfx["tax_total"], errors="coerce").fillna(0.0)
            dfx["total_ex_vat"] = (total_num - tax).clip(lower=0.0)
        elif "Taxes" in dfx.columns:  # Excel export support
            tax = pd.to_numeric(dfx["Taxes"], errors="coerce").fillna(0.0)
            dfx["total_ex_vat"] = (total_num - tax).clip(lower=0.0)
        else:
            dfx["total_ex_vat"] = total_num.apply(ex_vat)

    # Apply any global exclusions if you defined them
    try:
        dfx = dfx[
            ~dfx["summary"].astype(str).str.contains(EXCLUDE_SUMMARY_RE, na=False)
            & ~dfx["status_name"].astype(str).str.contains(EXCLUDE_STATUS_RE, na=False)
        ]
    except NameError:
        pass

    # Status
    if status_filter != "All":
        dfx = dfx[dfx["status_name"] == status_filter]

    # Search
    s = (search or "").strip().lower()
    if s:
        dfx = dfx[dfx.apply(
            lambda r: s in str(r.get("customer_name", "")).lower()
                   or s in str(r.get("customer_email", "")).lower()
                   or s in str(r.get("code", "")).lower()
                   or s in str(r.get("booking_id", "")).lower(),
            axis=1
        )]

    # Category filter (and Tour allow-list enforcement)
    if category_filter != "All":
        dfx = dfx[dfx["product_category"] == category_filter]

        if category_filter == "Tour":
            # Keep rows where ANY part of the summary is an allowed tour
            dfx = dfx[dfx["summary"].astype(str).apply(contains_allowed_tour)]

    # Specific product (match against any part of a multi-item summary)
    if specific_product != "All":
        target = _norm_title(specific_product)

        def _has_target(summary: str) -> bool:
            parts = _parts(summary)          # split multi-item summaries
            return target in parts           # exact part match

        dfx = dfx[dfx["summary"].astype(str).apply(_has_target)]

    return dfx


# Normalized local day window for all booking/event filters (exclusive end bound)
start_ts = pd.Timestamp(start)                      # local 00:00
end_excl = pd.Timestamp(end) + pd.Timedelta(days=1) # next-day 00:00 (exclusive)

# --- Load booking-date dataset ---
raw_booking = get_raw(start, end, include_items=False, filter_on="created")
df_booking = prepare_df(raw_booking)
df_booking = _apply_filters(df_booking)
cd_booking = pd.to_datetime(df_booking["created_date"], errors="coerce")
mask_b = cd_booking.notna() & (cd_booking >= start_ts) & (cd_booking < end_excl)
view_booking = df_booking.loc[mask_b].copy()

with st.expander("🔎 Debug: Raw Bookings (before filters)"):
    raw_df = prepare_df(raw_booking).copy()
    st.dataframe(
        raw_df[[
            "booking_id", 
            "created_date", 
            "summary", 
            "status_name", 
            "customer_name", 
            "customer_email", 
            "total", 
            "tax_total"
        ]].sort_values("created_date", ascending=False),
        use_container_width=True
    )
    st.caption(f"Total raw rows before filters: {len(raw_df)}")


# --- Load event-date dataset (on demand) ---
@st.cache_data(ttl=300)
def get_event_df(start, end):
    raw_event = get_raw(start, end, include_items=True, filter_on="event")
    df_event = prepare_df(raw_event).copy()
    if "items" not in df_event.columns:
        if "item" in df_event.columns:
            def _to_list(x):
                if isinstance(x, dict): return list(x.values())
                if isinstance(x, list): return x
                return []
            df_event["items"] = df_event["item"].apply(_to_list)
        else:
            df_event["items"] = [[] for _ in range(len(df_event))]
    df_event["event_date"] = df_event["items"].apply(_extract_event_dt_from_items)
    if "date_desc" in df_event.columns:
        df_event["event_date"] = df_event["event_date"].fillna(pd.to_datetime(df_event["date_desc"], errors="coerce"))
    if "created_date" in df_event.columns:
        df_event["event_date"] = df_event["event_date"].fillna(df_event["created_date"])
    df_event["event_date"] = pd.to_datetime(df_event["event_date"], errors="coerce")
    return df_event

# Only define event view if needed
if date_basis == "Event date":
    df_event = get_event_df(start, end)
    df_event = _apply_filters(df_event)
    ed_event = pd.to_datetime(df_event["event_date"], errors="coerce")
    mask_e = ed_event.notna() & (ed_event >= start_ts) & (ed_event < end_excl)
    view_event = df_event.loc[mask_e].copy()
else:
    view_event = pd.DataFrame()

# Choose current view (drives KPIs & charts up to Stock section)
if date_basis == "Event date":
    current_view = view_event.copy()
    current_view["day"] = current_view["event_date"].dt.day_name()
    current_view["hour"] = current_view["event_date"].dt.hour
    basis_series = pd.to_datetime(current_view["event_date"], errors="coerce")
    basis_label = "Event"
else:
    current_view = view_booking.copy()
    basis_series = pd.to_datetime(current_view["created_date"], errors="coerce")
    basis_label = "Booking"
    
    

with st.expander("🧪 Diagnostics: Why are rows being excluded?", expanded=False):
    # Work from the *unfiltered* base for the chosen basis
    base_df = view_event.copy() if date_basis == "Event date" else view_booking.copy()

    # Make sure derived columns exist
    if "product_category" not in base_df.columns:
        base_df["product_category"] = base_df["summary"].apply(categorise_product)

    # Evaluate each rule as a boolean column
    s = (search or "").strip().lower()
    diag = base_df.copy()
    diag["ok_status"]   = diag.apply(_matches_status, axis=1)
    diag["ok_search"]   = diag.apply(lambda r: _matches_search(r, s), axis=1)
    diag["ok_category"] = diag.apply(lambda r: _matches_category(r), axis=1)
    diag["ok_tourlist"] = diag.apply(lambda r: _passes_tour_allow(r), axis=1)
    diag["ok_specific"] = diag.apply(lambda r: _matches_specific_product(r), axis=1)

    # Final decision & reason
    checks = ["ok_status", "ok_search", "ok_category", "ok_tourlist", "ok_specific"]
    diag["INCLUDED"] = diag[checks].all(axis=1)
    def _reason(r):
        if r["INCLUDED"]:
            return ""
        failed = [c for c in checks if not r[c]]
        return ", ".join(failed)
    diag["excluded_by"] = diag.apply(_reason, axis=1)

    # Summary counts
    total = len(diag)
    kept  = int(diag["INCLUDED"].sum())
    dropped = total - kept
    st.write(f"Total rows (basis={date_basis}): **{total}**  |  Included: **{kept}**  |  Dropped: **{dropped}**")

    # Show dropped rows with reasons
    st.markdown("**Dropped rows (with reasons):**")
    cols_to_show = ["booking_id","created_date","summary","status_name","customer_name","customer_email","total_ex_vat","excluded_by"]
    existing_cols = [c for c in cols_to_show if c in diag.columns]
    st.dataframe(
        diag.loc[~diag["INCLUDED"], existing_cols].sort_values(existing_cols[1] if len(existing_cols)>1 else "booking_id", ascending=False),
        use_container_width=True
    )

    # Show included sample (optional)
    st.markdown("**Included rows (sample):**")
    st.dataframe(diag.loc[diag["INCLUDED"], existing_cols].head(20), use_container_width=True)


if current_view.empty:
    st.warning("No bookings match this window for the selected date basis and filters.")
    st.stop()

with st.expander("🔎 Debug: All Bookings in Current View"):
    st.dataframe(
        current_view[[
            "booking_id", 
            "created_date", 
            "summary", 
            "status_name", 
            "customer_name", 
            "customer_email", 
            AMOUNT_COL
        ]].sort_values("created_date", ascending=False),
        use_container_width=True
    )
    st.caption(f"Total rows in current view: {len(current_view)}")



if st.sidebar.button("🔄 Force refresh data"):
    st.cache_data.clear()
    st.cache_resource.clear()

# --- KPIs (uses AMOUNT_COL everywhere) ---
total_bookings = len(current_view)
amount_series = pd.to_numeric(current_view.get(AMOUNT_COL, 0), errors="coerce").fillna(0)
total_amount  = amount_series.sum()
avg_booking   = (total_amount / total_bookings) if total_bookings else 0.0
paid_pct    = (pd.to_numeric(current_view.get("paid_total", 0), errors="coerce").fillna(0) > 0).mean() * 100
repeat_rate = current_view["customer_email"].duplicated().mean() * 100 if "customer_email" in current_view.columns else 0.0

kpi_data = {
    "Total Bookings": total_bookings,
    AMOUNT_LABEL: f"£{total_amount:,.2f}",
    "Avg per Booking": f"£{avg_booking:,.2f}",
    "Paid %": f"{paid_pct:.1f}%",
    "Repeat Customers %": f"{repeat_rate:.1f}%"
}


k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Bookings", total_bookings)
k2.metric(AMOUNT_LABEL, f"£{total_amount:,.2f}")
k3.metric("Avg per Booking (ex VAT)", f"£{avg_booking:,.2f}")
k4.metric("Paid %", f"{paid_pct:.1f}%")
k5.metric("Repeat Cust %", f"{repeat_rate:.1f}%")


# --- Charts ---
st.markdown("### 📈 Insights")

current_view = current_view.copy()
current_view["basis_date"] = basis_series.dt.date

col1, col2 = st.columns(2)
with col1:
    ts = (current_view.groupby("basis_date").size().reset_index(name="Bookings").sort_values("basis_date"))
    fig1 = px.line(ts, x="basis_date", y="Bookings", title=f"📅 Bookings Over Time ({basis_label} Date)")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    pie = current_view["status_name"].value_counts().reset_index()
    pie.columns = ["Status", "Count"]
    fig2 = px.pie(pie, names="Status", values="Count", title="📌 Booking Status Breakdown")
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    product_amount = (
        current_view.groupby("summary")[AMOUNT_COL]
        .sum()
        .reset_index()
        .sort_values(AMOUNT_COL, ascending=False)
        .rename(columns={AMOUNT_COL: AMOUNT_LABEL})
    )
    fig3 = px.bar(
        product_amount, x="summary", y=AMOUNT_LABEL,
        title=f"💰 {AMOUNT_LABEL} by Product ({basis_label} Date, Selected Range)", text_auto=True
    )
    fig3.update_yaxes(tickprefix="£", tickformat=",")
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    # === QUARTERLY COMPARISON (APRIL → MARCH), not affected by date filter ===
    today = date.today()
    if today.month < 4:
        fy_start = date(today.year - 1, 4, 1)
        fy_end   = date(today.year, 3, 31)
    else:
        fy_start = date(today.year, 4, 1)
        fy_end   = date(today.year + 1, 3, 31)

    if date_basis == "Event date":
        df_q = get_event_df(fy_start, fy_end)
        df_q = _apply_filters(df_q)
        date_series_q = pd.to_datetime(df_q["event_date"], errors="coerce")
    else:
        raw_q = get_raw(fy_start, fy_end, include_items=False, filter_on="created")
        df_q = _apply_filters(prepare_df(raw_q))
        date_series_q = pd.to_datetime(df_q["created_date"], errors="coerce")

    df_q = df_q[date_series_q.notna()].copy()
    df_q["periodQ"] = date_series_q.dt.to_period("Q")

    q_index  = pd.period_range(start=fy_start, end=fy_end, freq="Q")
    q_series = df_q.groupby("periodQ")[AMOUNT_COL].sum()
    q_df     = q_series.reindex(q_index, fill_value=0).reset_index()
    q_df.columns = ["quarter", "total"]

    q_df.loc[(q_df["quarter"].dt.start_time > pd.Timestamp.today()), "total"] = pd.NA

    def _fy_quarter_label(p):
        m = p.start_time.month
        return {4: "Q1 (Apr–Jun)", 7: "Q2 (Jul–Sep)", 10: "Q3 (Oct–Dec)", 1: "Q4 (Jan–Mar)"}\
               .get(m, str(p))
    q_df["quarter"] = q_df["quarter"].apply(_fy_quarter_label)

    fy_label = f"FY {fy_start.year}/{fy_end.year}"

    fig_quarter = px.bar(
        q_df, x="quarter", y="total",
        title=f"Quarterly {AMOUNT_LABEL} Comparison ({basis_label} Date, {fy_label})",
        text="total"
    )
    fig_quarter.update_traces(texttemplate="£%{y:,.0f}")
    fig_quarter.update_yaxes(tickprefix="£", tickformat=",")
    st.plotly_chart(fig_quarter, use_container_width=True)

# ----------------  MONTHLY (full-month totals for months touched by range)  ----------------
m_start = date(start.year, start.month, 1)
m_end   = date(end.year, end.month, calendar.monthrange(end.year, end.month)[1])

if date_basis == "Event date":
    df_m  = get_event_df(m_start, m_end)
    df_m  = _apply_filters(df_m)
    date_series_m = pd.to_datetime(df_m["event_date"], errors="coerce")
else:
    raw_m = get_raw(m_start, m_end, include_items=False, filter_on="created")
    df_m  = _apply_filters(prepare_df(raw_m))
    date_series_m = pd.to_datetime(df_m["created_date"], errors="coerce")

df_m = df_m[date_series_m.notna()].copy()
df_m["periodM"] = date_series_m.dt.to_period("M")

month_idx = pd.period_range(start=m_start, end=m_end, freq="M")
m_series  = df_m.groupby("periodM")[AMOUNT_COL].sum().sort_index()
monthly   = m_series.reindex(month_idx, fill_value=0).reset_index()
monthly.columns = ["Month", AMOUNT_LABEL]
monthly["Month"] = monthly["Month"].dt.strftime("%b %Y")

fig_monthly = px.bar(
    monthly, x="Month", y=AMOUNT_LABEL,
    title=f"📆 Total Monthly {AMOUNT_LABEL} ({basis_label} Date scope)",
    text=AMOUNT_LABEL
)
fig_monthly.update_traces(texttemplate="£%{y:,.0f}")
fig_monthly.update_yaxes(tickprefix="£", tickformat=",")
st.plotly_chart(fig_monthly, use_container_width=True)

# Day/Hour charts
c5, c6 = st.columns(2)

with c5:
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    day_counts = current_view["day"].value_counts().reindex(day_order).fillna(0).reset_index()
    day_counts.columns = ["Day", "Count"]
    st.plotly_chart(
        px.bar(day_counts, x="Day", y="Count",
               title=f"📅 Bookings by Day ({basis_label} Date)"),
        use_container_width=True
    )

with c6:
    if date_basis != "Event date":
        hour_counts = (
            pd.to_datetime(current_view["created_date"], errors="coerce")
              .dt.hour.value_counts().sort_index().reset_index()
        )
        hour_counts.columns = ["Hour", "Count"]
        st.plotly_chart(
            px.bar(hour_counts, x="Hour", y="Count",
                   title="⏰ Bookings by Hour (Created Time)"),
            use_container_width=True
        )
    else:
        st.caption("⏰ Hour breakdown is hidden for Event date (events typically have no time).")

# === Extended comparisons (weekly, using week beginning dates) ===
def render_extended_time_comparisons(view_df: pd.DataFrame, basis_series: pd.Series, basis_label: str,
                                     amount_col: str, amount_label: str):
    dfv = view_df.copy()

    dfv["_basis_ts"] = pd.to_datetime(basis_series, errors="coerce")
    dfv = dfv[dfv["_basis_ts"].notna()].copy()

    dfv["week_start"] = dfv["_basis_ts"].dt.to_period("W").apply(lambda r: r.start_time.normalize())
    dfv["week_label"] = dfv["week_start"].dt.strftime("%d %b %Y")
    dfv["year"] = dfv["_basis_ts"].dt.year

    st.markdown("### 📊 Extended Time-Based Comparisons")

    # 1) Weekly Breakdown by Product (Amount or ex-VAT)
    st.subheader(f"Weekly Breakdown by Product — {basis_label} date (week beginning)")
    weekly_breakdown = (
        dfv.groupby(["week_start", "week_label", "summary"], as_index=False)[amount_col]
           .sum()
           .sort_values("week_start")
           .rename(columns={amount_col: amount_label})
    )
    fig_wbp = px.bar(
        weekly_breakdown,
        x="week_label", y=amount_label, color="summary", barmode="group",
        title=f"Weekly Breakdown by Product ({amount_label})"
    )
    fig_wbp.update_yaxes(tickprefix="£", tickformat=",")
    ordered_labels = weekly_breakdown.drop_duplicates("week_label")["week_label"].tolist()
    fig_wbp.update_xaxes(categoryorder="array", categoryarray=ordered_labels)
    st.plotly_chart(fig_wbp, use_container_width=True)

    # 2) Weekly Amount/Ex-VAT by Year
    st.subheader(f"Weekly {amount_label} by Year — {basis_label} date (week beginning)")
    weekly_compare = (
        dfv.groupby(["year", "week_start", "week_label"], as_index=False)[amount_col]
           .sum()
           .sort_values("week_start")
           .rename(columns={amount_col: amount_label})
    )
    fig_wc = px.line(
        weekly_compare,
        x="week_label", y=amount_label, color="year", markers=True,
        title=f"Weekly {amount_label} by Year"
    )
    fig_wc.update_yaxes(tickprefix="£", tickformat=",")
    ordered_labels_wc = weekly_compare.drop_duplicates("week_label")["week_label"].tolist()
    fig_wc.update_xaxes(categoryorder="array", categoryarray=ordered_labels_wc)
    st.plotly_chart(fig_wc, use_container_width=True)

# Call extended comparisons
render_extended_time_comparisons(current_view, basis_series, basis_label, AMOUNT_COL, AMOUNT_LABEL)

# === STOCK AVAILABILITY & MISSED REVENUE ===
st.markdown("##  Stock Availability & Missed Revenue")
st.caption("Filtered to tours only. Ticket totals are the number of tickets sold (EVENT-based).")

stock_start = st.date_input("Start Date for Stock Analysis", value=date.today())
stock_end   = st.date_input("End Date for Stock Analysis",   value=date.today() + timedelta(days=30))
num_days = (stock_end - stock_start).days + 1

try:
    # Pull bookings for THIS window by EVENT DATE (expand items)
    try:
        raw_stock = get_raw(stock_start, stock_end, include_items=True, filter_on="event")
    except TypeError:
        raw_stock = get_raw(stock_start, stock_end, include_items=True)

    df_items = prepare_df(raw_stock).copy()

    # Ensure 'items' exists and is a list
    if "items" not in df_items.columns:
        if "item" in df_items.columns:
            def _to_list(x):
                if isinstance(x, dict): return list(x.values())
                if isinstance(x, list): return x
                return []
            df_items["items"] = df_items["item"].apply(_to_list)
        else:
            df_items["items"] = [[] for _ in range(len(df_items))]

    def _safe_int(v):
        try:
            n = int(float(v));  return n if n >= 0 else 0
        except Exception:
            return 0

    # Ticket qty from items; fallback to /booking/{id} if mostly zeros
    def _count_qty(items):
        if isinstance(items, dict): items = list(items.values())
        if not isinstance(items, list): return 0
        t = 0
        for it in items:
            if isinstance(it, dict): t += _safe_int(it.get("qty", 0))
        return t

    df_items["ticket_qty"] = df_items["items"].apply(_count_qty).fillna(0).astype(int)

    if len(df_items) > 0 and (df_items["ticket_qty"] == 0).mean() > 0.9:
        enriched = []
        for _, row in df_items.iterrows():
            bid = row.get("booking_id")
            if not bid:
                enriched.append(0); continue
            try:
                d = fetch_booking_details(bid)
                items = (d.get("booking", {}) or {}).get("items", [])
                if isinstance(items, dict): items = list(items.values())
                qty = 0
                if isinstance(items, list):
                    for it in items:
                        if isinstance(it, dict): qty += _safe_int(it.get("qty", 0))
                enriched.append(qty)
            except Exception:
                enriched.append(0)
        df_items["ticket_qty"] = enriched

    # Robust event_date extraction
    df_items["event_date"] = df_items["items"].apply(_extract_event_dt_from_items)
    if "date_desc" in df_items.columns:
        df_items["event_date"] = df_items["event_date"].fillna(pd.to_datetime(df_items["date_desc"], errors="coerce"))
    if "created_date" in df_items.columns:
        df_items["event_date"] = df_items["event_date"].fillna(df_items["created_date"])
    df_items["event_date"] = pd.to_datetime(df_items["event_date"], errors="coerce")

    # Filter by EVENT DATE + tours only
    df_stock_base = df_items.loc[
        (df_items["event_date"] >= pd.Timestamp(stock_start)) &
        (df_items["event_date"] <= pd.Timestamp(stock_end))
    ].copy()

    df_stock_base["summary"] = df_stock_base["summary"].astype(str).str.strip()
    is_tour = (
        df_stock_base["summary"].str.contains(r"\btour\b", case=False, na=False) &
        ~df_stock_base["summary"].str.contains(r"voucher|gift|guidebook|souvenir|meeting|room|hire", case=False, na=False)
    )
    df_stock_base = df_stock_base[is_tour].copy()

    # Observed unit prices (always ex VAT)
    product_prices = {}
    df_tmp = df_stock_base[df_stock_base["ticket_qty"].fillna(0) > 0].copy()
    if not df_tmp.empty:
        df_tmp["unit_price"] = df_tmp.apply(
            lambda r: ((r["total"] - r.get("Taxes", 0)) / r["ticket_qty"]) if r["ticket_qty"] else 0.0,
            axis=1
        )
        product_prices = df_tmp.groupby("summary")["unit_price"].median().to_dict()


    # ---------- Capacity & departures ----------
    TOUR_SEATS_PER_DEP = 12  # business rule

    perday_tickets = (
        df_stock_base
        .assign(_day=df_stock_base["event_date"].dt.floor("D"))
        .groupby(["summary", "_day"], as_index=False)["ticket_qty"]
        .sum()
    )
    perday_tickets["deps_day"] = (perday_tickets["ticket_qty"] + TOUR_SEATS_PER_DEP - 1) // TOUR_SEATS_PER_DEP

    deps_per_day_est = perday_tickets.groupby("summary")["deps_day"].mean().round(2).to_dict()
    days_with_deps   = perday_tickets.groupby("summary")["deps_day"].count().to_dict()
    total_deps       = perday_tickets.groupby("summary")["deps_day"].sum().to_dict()

    # Debug table
    debug_rows = []
    for product in sorted(df_stock_base["summary"].dropna().unique()):
        debug_rows.append({
            "Product": product,
            "Seats per Departure (detected)": TOUR_SEATS_PER_DEP,
            "Capacity Source": "rule-fixed-12",
            "Avg Departures/Day": float(deps_per_day_est.get(product, 0.0)),
            "Days with Departures": int(days_with_deps.get(product, 0)),
            "Total Departures Observed": int(total_deps.get(product, 0)),
        })
    debug_df = pd.DataFrame(debug_rows).sort_values(["Product"]).reset_index(drop=True)

    with st.expander("🔎 Capacity & Departures (debug)"):
        st.dataframe(debug_df)

    # --- Build Stock table ---
    rows = []
    cap_sources = {}

    price_col_label = "Price (ex VAT) (£)"
    lost_col_label  = "Potential Revenue Lost (ex VAT) (£)"


    for product in sorted(df_stock_base["summary"].dropna().unique()):
        pdfp = df_stock_base[df_stock_base["summary"] == product]
        tickets_booked = int(pdfp["ticket_qty"].fillna(0).sum())

        seats_per_dep = TOUR_SEATS_PER_DEP
        cap_sources[product] = "rule-fixed-12"

        avg_deps_per_day = float(deps_per_day_est.get(product, 0.0))
        total_capacity = int(round(seats_per_dep * avg_deps_per_day * num_days))
        available = max(total_capacity - tickets_booked, 0)

        avg_price = float(product_prices.get(product, 0.0))
        lost_revenue = available * avg_price

        rows.append({
            "Product": product,
            "Booked Tickets": tickets_booked,
            "Seats/Departure (detected)": seats_per_dep,
            "Avg Departures/Day": round(avg_deps_per_day, 2),
            "Capacity (Seats x Deps x Days)": total_capacity,
            "Available": available,
            price_col_label: round(avg_price, 2),
            lost_col_label: round(lost_revenue, 2),
        })

    stock_df = pd.DataFrame(rows)

    with st.expander("🔎 Debug (stock)"):
        st.write("Capacity source by product:", cap_sources)
        st.write("Products in window:", sorted(df_stock_base["summary"].dropna().unique().tolist()))
        st.write("Rows with non-zero tickets:", int((df_stock_base["ticket_qty"] > 0).sum()))
        st.write("Sample stock rows:", stock_df.head(10))

    # --- Render table + charts ---
    with st.expander("📋 Full Stock & Revenue Table"):
        st.dataframe(stock_df)

    left, right = st.columns(2)
    with left:
        fig_stock = px.bar(
            stock_df,
            x="Product",
            y=["Booked Tickets", "Available"],
            barmode="stack",
            title="🎟️ Stock vs Tickets Booked (avg deps/day × days, 12 seats/dep)"
        )
        st.plotly_chart(fig_stock, use_container_width=True)

    with right:
        lost_sorted = stock_df.sort_values(lost_col_label, ascending=False).copy()
        fig_lost = px.bar(
            lost_sorted,
            x=lost_col_label,
            y="Product",
            orientation="h",
            title=f"💸 {lost_col_label} by Product",
            text=lost_col_label
        )
        fig_lost.update_yaxes(categoryorder="array", categoryarray=list(lost_sorted["Product"]))
        fig_lost.update_xaxes(tickprefix="£", tickformat=",")
        fig_lost.update_traces(
            texttemplate="£%{x:,.0f}",
            hovertemplate="<b>%{y}</b><br>Lost: £%{x:,.2f}<extra></extra>"
        )
        st.plotly_chart(fig_lost, use_container_width=True)

except Exception as e:
    st.warning("⚠️ Error calculating stock and lost revenue.")
    st.error(str(e))

# --- PDF Download (booking-date based view for recent rows/top tour; respects VAT toggle) ---
top_tour = view_booking.groupby("summary")[AMOUNT_COL].sum().idxmax() if not view_booking.empty else "N/A"
top_day = view_booking["day"].mode()[0] if not view_booking.empty and "day" in view_booking.columns else "N/A"
recent_rows = view_booking.sort_values("created_date", ascending=False).head(5).to_dict(orient="records")
date_range = f"{start.strftime('%d %b %Y')} to {end.strftime('%d %b %Y')}"
pdf_bytes = create_detailed_pdf_summary(kpi_data, date_range, top_tour, top_day, recent_rows)
today_str = datetime.today().strftime("%Y-%m-%d")
pdf_filename = f"shoebox_summary_{today_str}.pdf"

st.sidebar.download_button(label="⬇️ Download PDF", data=pdf_bytes, file_name=pdf_filename, mime="application/pdf")







