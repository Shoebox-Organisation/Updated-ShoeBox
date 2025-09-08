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
from fpdf import FPDF
from pathlib import Path
from datetime import datetime

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
).rstrip("/")

# Derived endpoints (use these throughout)
BOOKING_INDEX_URL = f"{API_BASE}/index"
BOOKING_DETAIL_URL = API_BASE  # + f"/{{booking_id}}"

# Optional: temporary escape hatch in cloud (ONLY for short-term debugging)
ALLOW_INSECURE = str(_get_secret("ALLOW_INSECURE_SSL", "false")).lower() == "true"

# (Only needed if you ever switch to api.checkfront.com host)
CHECKFRONT_ACCOUNT = _get_secret("CHECKFRONT_ACCOUNT", "theshoebox")  # e.g., "theshoebox"


# --- TLS trust setup (Windows/macOS-friendly, works in cloud) ---
USING_OS_TRUST = False
try:
    import truststore
    truststore.inject_into_ssl()  # Use system trust store
    USING_OS_TRUST = True
except Exception:
    # Fallback: use certifi bundle (Linux/cloud)
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

# --- VAT config ---
VAT_RATE = 0.20


def ex_vat(amount: float | int | None, rate: float = VAT_RATE) -> float:
    """Fallback ex-VAT calculation when tax_total is unavailable."""
    try:
        return float(amount) / (1.0 + rate)
    except Exception:
        return 0.0


def _norm_title(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).strip().lower()
    s = re.sub(r"\s+", " ", s)      # collapse spaces
    s = s.replace("–", "-")         # en-dash → hyphen
    return s


SPLIT_RE = re.compile(r"\s*(?:,|/|&|\+| and )\s*")
def _parts(summary: str) -> list[str]:
    s = _norm_title(summary)
    if not s:
        return []
    return [p for p in SPLIT_RE.split(s) if p]


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

# ---- HTTPS session (retries, timeout, TLS verify) ----
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
        s.verify = certifi.where()  # Linux/cloud path

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
    if "api.checkfront.com" in API_BASE:
        h["X-Checkfront-Account"] = CHECKFRONT_ACCOUNT
    return h

# =========================
# HARD-CODED CATALOG + SIDEBAR (LIVE-CASCADING)
# =========================

def build_hardcoded_catalog() -> dict:
    CATALOG_MAP: dict[str, list[str]] = {
        "Norwich History Festival": [
            "Norwich History Festival – The Magnificent Marble Hall & The Bignolds",
            "Norwich History Festival – Rebels and Radicals",
        ],
        "Community Events": [],
        "Great Yarmouth": [
            "Great Yarmouth – Heritage Walk",
            "Great Yarmouth - Seafront Tour",
        ],
        "Norwich’s Hidden Street": [
            "Norwich’s Hidden Street Tour",
            "Norwich’s Hidden Street Tour – Family Fun",
        ],
        "Norwich Walking Tours": [
            "The TIPSY Tavern Trail Tour",
            "The Tavern Trail Tour",
            "The Norwich Knowledge Tour",
            "City of Centuries Tour",
            "The Matriarchs, Mayors & Merchants Tour",
        ],
        "Escape Game": [],
        "Marble Hall": ["Magnificent Marble Hall"],
        "Valentine’s": ["Love Stories (With A Twist)"],
        "Lantern Light Underground Tour": ["Lantern Light Underground Tour"],
        "'Meet The Experience Hosts' Event": ["'Meet The Experience Hosts' Event"],
        "After Dark Experience": [],
        "Meeting Room Hire": [],
        "Community Hub Hire": [],
        "Room Hire Add Ons": [],
        "Gift Shop": [],
        "Christmas": ["The Chronicles of Christmas"],
        "Archive": [],
        "Booking Amendment Charges": [
            "Late reschedule or cancellation (non COVID related)",
            "Late reschedule or cancellation (COVID related)",
        ],
        "Capacity Control": [],
        "Gift Vouchers": ["eGift Voucher for The Shoebox Experiences"],
    }

    cat_to_products: dict[str, list[str]] = {c: sorted(set(v)) for c, v in CATALOG_MAP.items()}
    all_categories = sorted(cat_to_products.keys())

    name_to_cats: dict[str, set[str]] = {}
    for cat, items in cat_to_products.items():
        for it in items:
            name_to_cats.setdefault(_norm_title(it), set()).add(cat)

    return {
        "cat_to_products": cat_to_products,
        "all_categories": all_categories,
        "name_to_cats": name_to_cats,
        "cat_id_by_name": {},
        "item_name_to_id": {},
    }

catalog = build_hardcoded_catalog()

# Session defaults for cascading widgets + submit gating
if "cat_ms" not in st.session_state:  st.session_state.cat_ms = []
if "item_ms" not in st.session_state: st.session_state.item_ms = []
if "applied_once" not in st.session_state: st.session_state.applied_once = False

# =========================
# Header + Quick Help + Sidebar (EARLY GATE)
# =========================

# --- Header ---
st.markdown("<h1 style='text-align: center;'>Shoebox Internal Operations Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- Quick Help (top banner) ---
help_box = st.empty()
if not st.session_state.get("applied_once", False):
    help_box.info(
        "First click **Apply filters** in the sidebar on the left. "
        "Then set your filters and press **Apply filters** once more."
    )
else:
    help_box.caption("Tip: adjust filters any time in the sidebar and click **Apply filters**.")

# --- Sidebar ---
with st.sidebar:
    logo = Path(__file__).parent / "shoebox.png"
    if logo.exists():
        st.image(str(logo), width=180)

    today = date.today()
    start = st.date_input("Start Date", today - timedelta(days=30), key="start_date")
    end   = st.date_input("End Date",   today + timedelta(days=60), key="end_date")
    search = st.text_input("🔍 Search name, email or booking code", key="search_text").strip()

    date_basis = st.selectbox(
        "Date basis for KPIs & charts",
        ["Booking date (created)", "Event date"],
        index=0,
        key="date_basis"
    )

    # Hard-coded catalog already built earlier
    cat_to_products = catalog["cat_to_products"]
    all_categories  = catalog["all_categories"]

    selected_categories = st.multiselect(
        "Category",
        options=all_categories,
        default=st.session_state.cat_ms,
        key="cat_ms",
        help="Choose one or more categories"
    )

    product_pool = sorted({
        p for c in (selected_categories or all_categories)
        for p in cat_to_products.get(c, [])
    })

    # prune invalid selections when category changes
    st.session_state.item_ms = [p for p in st.session_state.item_ms if p in product_pool]

    selected_products = st.multiselect(
        "Items (in selected categories)",
        options=product_pool,
        default=st.session_state.item_ms,
        key="item_ms",
        help="Only items from the selected categories appear here"
    )

    # Apply / Reset
    c1, c2 = st.columns(2)
    apply_clicked = c1.button("Apply filters", use_container_width=True)
    reset_clicked = c2.button("Reset", type="secondary", use_container_width=True)

    if reset_clicked:
        st.session_state.cat_ms = []
        st.session_state.item_ms = []
        st.session_state.search_text = ""
        st.session_state.status_sel = "All"
        st.session_state.applied_once = False
        st.rerun()

    if apply_clicked:
        st.session_state.applied_once = True

# --- EARLY GATE (very important): do not fetch data until Apply has been clicked once ---
if not st.session_state.get("applied_once", False):
    st.stop()

# From here on it's safe to do any data fetching (get_raw, etc.)

# Build status choices only AFTER the gate, so first-load shows the help instead of a spinner
temp_raw = get_raw(start, end, include_items=False, filter_on="created")
temp_df  = prepare_df(temp_raw)
status_choices = ["All"] + (sorted(temp_df["status_name"].dropna().unique()) if not temp_df.empty else [])

with st.sidebar:
    status_filter = st.selectbox("Filter by Booking Status", status_choices, index=0, key="status_sel")


# Mirror old variables
start = st.session_state.start_date
end = st.session_state.end_date
search = st.session_state.search_text
date_basis = st.session_state.date_basis
selected_categories = st.session_state.cat_ms
selected_products = st.session_state.item_ms

# Gate the app like the old form submit
if not st.session_state.applied_once and "data_ready" not in st.session_state:
    st.stop()
st.session_state.data_ready = True

# --- Auth & API helpers using API_BASE ---
def fetch_bookings(start_date, end_date, limit=250, include_items=False, max_pages=None,
                   filter_on="created", category_ids=None, item_ids=None):
    """Fetch all pages (newest-first)."""
    headers = get_auth_header()
    category_ids = list(category_ids or [])
    item_ids = list(item_ids or [])

    def _get(params):
        r = SESSION.get(BOOKING_INDEX_URL, headers=headers, params=params)
        if not r.ok:
            msg = f"{r.status_code} {r.reason} — {r.url}"
            st.error(f"API error: {msg}\n\n{r.text[:500]}")
            r.raise_for_status()
        return r.json()

    common = {"limit": limit, "sort": "created_date", "dir": "desc"}
    if include_items:               common["expand"]        = "items"
    if category_ids:                common["category_id[]"] = category_ids
    if item_ids:                    common["item_id[]"]     = item_ids

    # Try server-side windows in this order, else fallback to newest-first:
    def _params_for(field):
        if filter_on == "created":
            return common | {f"{field}[min]": start_date.isoformat(),
                             f"{field}[max]": end_date.isoformat()}
        else:
            return common | {"start_date[min]": start_date.isoformat(),
                             "start_date[max]": end_date.isoformat()}

    param_options = [
        _params_for("created_at"),
        _params_for("created_date"),
        common,
    ]

    def _page_all(params):
        out, seen, page = [], set(), 1
        while True:
            q = params.copy(); q["page"] = page
            data = _get(q)
            rows = list((data.get("booking/index") or {}).values())
            if not rows:
                break
            for b in rows:
                bid = b.get("booking_id")
                if bid and bid not in seen:
                    seen.add(bid); out.append(b)
            if len(rows) < params["limit"]:
                break
            page += 1
            if max_pages is not None and page > max_pages:
                break
        return out

    rows = []
    for params in param_options:
        try:
            rows = _page_all(params)
            break
        except requests.HTTPError as e:
            if not (e.response is not None and e.response.status_code in (400, 403)):
                raise

    # Client-side created-date filter as a safety net
    def _created_ts(b):
        for k in ("created_at", "created_date", "created", "date_created", "timestamp_created"):
            v = b.get(k)
            if v is None: continue
            ts = pd.to_datetime(v, unit="s", errors="coerce")
            if pd.isna(ts): ts = pd.to_datetime(v, errors="coerce")
            if pd.notna(ts):
                try:
                    return (ts.tz_localize("UTC").tz_convert(TENANT_TZ).tz_localize(None))
                except Exception:
                    return ts
        return None

    filtered = []
    for b in rows:
        ts = _created_ts(b)
        if ts is None or start_date <= ts.date() <= end_date:
            filtered.append(b)

    return {"booking/index": {i: b for i, b in enumerate(filtered)}}


def fetch_booking_details(booking_id: str | int):
    url = f"{BOOKING_DETAIL_URL}/{booking_id}"
    r = SESSION.get(url, headers=get_auth_header(), params={"expand": "items"}, timeout=15)
    if not r.ok:
        msg = f"{r.status_code} {r.reason} — {r.url}"
        st.error(f"API error: {msg}\n\n{r.text[:500]}")
        r.raise_for_status()
    return r.json()

# --- Cache API results ---
@st.cache_data(ttl=300)
def get_raw(start, end, include_items=False, filter_on="created",
            category_ids=None, item_ids=None):
    cat_key  = tuple(sorted(category_ids or []))
    item_key = tuple(sorted(item_ids or []))
    return fetch_bookings(start, end, include_items=include_items, filter_on=filter_on,
                          category_ids=cat_key, item_ids=item_key)

# --- (Legacy) Categorisation helper (still populates product_category for reference) ---
def categorise_product(summary: str) -> str:
    ns = _norm_title(summary)
    if not ns:
        return "Other"
    if any(p in TOURS_ALLOWLIST for p in _parts(summary)):
        return "Tour"
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

    # created_date: normalise to Europe/London
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

    for col in ("total", "tax_total"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["status_name"] = df.get("status_name", "Unknown").fillna("Unknown")
    df["summary"] = df.get("summary", "").astype(str).str.strip()
    df["day"]  = df["created_date"].dt.day_name()
    df["hour"] = df["created_date"].dt.hour

    df["product_category"] = df["summary"].apply(categorise_product)

    if "tax_total" in df.columns:
        tax = pd.to_numeric(df["tax_total"], errors="coerce").fillna(0.0)
        df["total_ex_vat"] = (df["total"].fillna(0.0) - tax).clip(lower=0.0)
    elif "Taxes" in df.columns:
        tax = pd.to_numeric(df["Taxes"], errors="coerce").fillna(0.0)
        df["total_ex_vat"] = (df["total"].fillna(0.0) - tax).clip(lower=0.0)
    else:
        df["total_ex_vat"] = df["total"].fillna(0.0).apply(ex_vat)

    if "booking_id" in df.columns:
        df = df.drop_duplicates(subset="booking_id", keep="last")
    else:
        df = df.drop_duplicates()

    return df

# ---------- Client-side filters (unchanged behaviour) ----------
def _apply_filters(dfx: pd.DataFrame,
                   selected_categories: list[str],
                   selected_products: list[str],
                   status_filter: str,
                   search: str) -> pd.DataFrame:
    dfx = dfx.copy()

    if "summary" not in dfx.columns:     dfx["summary"] = ""
    if "status_name" not in dfx.columns: dfx["status_name"] = "Unknown"
    if "total_ex_vat" not in dfx.columns:
        total_num = pd.to_numeric(dfx.get("total", 0.0), errors="coerce").fillna(0.0)
        if "tax_total" in dfx.columns:
            tax = pd.to_numeric(dfx["tax_total"], errors="coerce").fillna(0.0)
            dfx["total_ex_vat"] = (total_num - tax).clip(lower=0.0)
        elif "Taxes" in dfx.columns:
            tax = pd.to_numeric(dfx["Taxes"], errors="coerce").fillna(0.0)
            dfx["total_ex_vat"] = (total_num - tax).clip(lower=0.0)
        else:
            dfx["total_ex_vat"] = total_num.apply(ex_vat)

    if status_filter != "All":
        dfx = dfx[dfx["status_name"] == status_filter]

    s = (search or "").strip().lower()
    if s:
        dfx = dfx[dfx.apply(
            lambda r: s in str(r.get("customer_name", "")).lower()
                   or s in str(r.get("customer_email", "")).lower()
                   or s in str(r.get("code", "")).lower()
                   or s in str(r.get("booking_id", "")).lower(),
            axis=1
        )]

    if selected_categories:
        sel_cats = set(selected_categories)
        def _has_selected_cat(summary: str) -> bool:
            for part in _parts(summary):
                cats = catalog["name_to_cats"].get(part, set())
                if cats & sel_cats:
                    return True
            return False
        dfx = dfx[dfx["summary"].astype(str).apply(_has_selected_cat)]

    if selected_products:
        selected_norm = {_norm_title(p) for p in selected_products}
        def _has_selected_product(summary: str) -> bool:
            return any(part in selected_norm for part in _parts(summary))
        dfx = dfx[dfx["summary"].astype(str).apply(_has_selected_product)]

    return dfx

# ---------- Load datasets (same layout/logic) ----------
# Build status choices now that functions exist
temp_raw = get_raw(start, end, include_items=False, filter_on="created")
temp_df  = prepare_df(temp_raw)
status_choices = ["All"] + (sorted(temp_df["status_name"].dropna().unique())
                            if not temp_df.empty else [])
# put in sidebar placeholder (created above)
with st.sidebar:
    status_filter = st.selectbox("Filter by Booking Status", status_choices, index=0, key="status_sel")

AMOUNT_COL   = "total_ex_vat"
AMOUNT_LABEL = "Amount (ex VAT)"

start_ts = pd.Timestamp(start)
end_excl = pd.Timestamp(end) + pd.Timedelta(days=1)

raw_booking = get_raw(start, end, include_items=False, filter_on="created",
                      category_ids=[], item_ids=[])
df_booking  = prepare_df(raw_booking)
df_booking  = _apply_filters(df_booking, selected_categories, selected_products, status_filter, search)

cd_booking  = pd.to_datetime(df_booking["created_date"], errors="coerce")
mask_b      = cd_booking.notna() & (cd_booking >= start_ts) & (cd_booking < end_excl)
view_booking = df_booking.loc[mask_b].copy()

# Event-basis (if used)
if date_basis == "Event date":
    raw_event = get_raw(start, end, include_items=True, filter_on="event",
                        category_ids=[], item_ids=[])
    df_event  = prepare_df(raw_event)
    if "items" not in df_event.columns:
        df_event["items"] = [[] for _ in range(len(df_event))]
    # Robust extraction
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

    df_event["event_date"] = df_event["items"].apply(_extract_event_dt_from_items)
    if "date_desc" in df_event.columns:
        df_event["event_date"] = df_event["event_date"].fillna(pd.to_datetime(df_event["date_desc"], errors="coerce"))
    df_event["event_date"] = pd.to_datetime(df_event["event_date"], errors="coerce")

    df_event  = _apply_filters(df_event, selected_categories, selected_products, status_filter, search)
    ed_event  = pd.to_datetime(df_event["event_date"], errors="coerce")
    mask_e    = ed_event.notna() & (ed_event >= start_ts) & (ed_event < end_excl)
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


# --- KPIs ---
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

# --- Charts (same placement as your original) ---
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

# ----------------  MONTHLY (full-month totals for months touched by range)  ----------------
m_start = date(start.year, start.month, 1)
m_end   = date(end.year, end.month, calendar.monthrange(end.year, end.month)[1])

if date_basis == "Event date":
    df_m  = df_event.copy()
    if df_m.empty:
        raw_m = get_raw(m_start, m_end, include_items=True, filter_on="event")
        df_m  = prepare_df(raw_m)
        if "items" not in df_m.columns:
            df_m["items"] = [[] for _ in range(len(df_m))]
        df_m["event_date"] = df_m["items"].apply(lambda it: pd.NaT)  # already computed earlier when used
        df_m["event_date"] = pd.to_datetime(df_m["event_date"], errors="coerce")
    df_m  = _apply_filters(df_m, selected_categories, selected_products, status_filter, search)
    date_series_m = pd.to_datetime(df_m["event_date"], errors="coerce")
else:
    raw_m = get_raw(m_start, m_end, include_items=False, filter_on="created")
    df_m  = _apply_filters(prepare_df(raw_m), selected_categories, selected_products, status_filter, search)
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

# === Extended comparisons (weekly etc.) — keep your existing function & placement if present ===
def render_extended_time_comparisons(view_df: pd.DataFrame, basis_series: pd.Series, basis_label: str,
                                     amount_col: str, amount_label: str):
    dfv = view_df.copy()
    dfv["_basis_ts"] = pd.to_datetime(basis_series, errors="coerce")
    dfv = dfv[dfv["_basis_ts"].notna()].copy()

    dfv["week_start"] = dfv["_basis_ts"].dt.to_period("W").apply(lambda r: r.start_time.normalize())
    dfv["week_label"] = dfv["week_start"].dt.strftime("%d %b %Y")
    dfv["year"] = dfv["_basis_ts"].dt.year

    st.markdown("### 📊 Extended Time-Based Comparisons")
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

# Call extended comparisons (same placement as your file)
render_extended_time_comparisons(current_view, basis_series, basis_label, AMOUNT_COL, AMOUNT_LABEL)

# === QUARTERLY COMPARISON (APRIL → MARCH), not affected by the selected range ===
st.markdown("### 📊 Quarterly Revenue (FY Apr–Mar) (Not affected by Filters")
subtitle = "Not affected by filters, however will show entire year of the filtered data"

# Financial year covering today, Apr -> Mar
today = date.today()
if today.month < 4:
    fy_start = date(today.year - 1, 4, 1)
    fy_end   = date(today.year, 3, 31)
else:
    fy_start = date(today.year, 4, 1)
    fy_end   = date(today.year + 1, 3, 31)

if date_basis == "Event date":
    # Pull event-dated bookings with items so we can derive event_date
    raw_q = get_raw(fy_start, fy_end, include_items=True, filter_on="event")
    df_q = prepare_df(raw_q).copy()
    if "items" not in df_q.columns:
        df_q["items"] = [[] for _ in range(len(df_q))]

    # Local helper to extract event datetime (robust)
    def _extract_event_dt_q(items):
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
                    dt = (pd.to_datetime(v, unit="s", errors="coerce")
                          if isinstance(v,(int,float)) else pd.to_datetime(v, errors="coerce"))
                    if pd.notna(dt): cands.append(dt)
            v = it.get("date_desc")
            if v is not None:
                dt = pd.to_datetime(v, errors="coerce")
                if pd.notna(dt): cands.append(dt)
        return min(cands) if cands else pd.NaT

    df_q["event_date"] = df_q["items"].apply(_extract_event_dt_q)
    if "date_desc" in df_q.columns:
        df_q["event_date"] = df_q["event_date"].fillna(pd.to_datetime(df_q["date_desc"], errors="coerce"))

    df_q = _apply_filters(df_q, selected_categories, selected_products, status_filter, search)
    date_series_q = pd.to_datetime(df_q["event_date"], errors="coerce")
else:
    # Booking-created basis
    raw_q = get_raw(fy_start, fy_end, include_items=False, filter_on="created")
    df_q = _apply_filters(prepare_df(raw_q), selected_categories, selected_products, status_filter, search)
    date_series_q = pd.to_datetime(df_q["created_date"], errors="coerce")

# Build quarter totals across the FY
df_q = df_q[date_series_q.notna()].copy()
df_q["periodQ"] = date_series_q.dt.to_period("Q")

q_index  = pd.period_range(start=fy_start, end=fy_end, freq="Q")
q_series = df_q.groupby("periodQ")[AMOUNT_COL].sum()
q_df     = q_series.reindex(q_index, fill_value=0).reset_index()
q_df.columns = ["quarter", "total"]

# Hide future quarters (beyond today)
q_df.loc[(q_df["quarter"].dt.start_time > pd.Timestamp.today()), "total"] = pd.NA

# Nice labels for Apr–Jun etc.
def _fy_quarter_label(p):
    m = p.start_time.month
    return {4: "Q1 (Apr–Jun)", 7: "Q2 (Jul–Sep)", 10: "Q3 (Oct–Dec)", 1: "Q4 (Jan–Mar)"}\
           .get(m, str(p))
q_df["quarter"] = q_df["quarter"].apply(_fy_quarter_label)

fy_label = f"FY {fy_start.year}/{fy_end.year}"
fig_quarter = px.bar(
    q_df, x="quarter", y="total",
    title=f"Quarterly {AMOUNT_LABEL} Comparison ({'Event' if date_basis=='Event date' else 'Booking'} Date, {fy_label})",
    text="total"
)
fig_quarter.update_traces(texttemplate="£%{y:,.0f}")
fig_quarter.update_yaxes(tickprefix="£", tickformat=",")
st.plotly_chart(fig_quarter, use_container_width=True)


# ================================
# Stock Availability + PDF Download
# ================================
from datetime import datetime, timedelta
from pathlib import Path
from fpdf import FPDF

# ---- PDF generator ----
def create_detailed_pdf_summary(kpi_data, date_range, top_tour, top_day, recent_rows,
                                logo_path=None, use_ex_vat=True):
    pdf = FPDF()
    pdf.add_page()

    # Optional logo
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

    # KPIs
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Key Metrics", ln=True)
    pdf.set_font("Arial", "", 12)
    for label, value in kpi_data.items():
        pdf.cell(0, 8, f"{label}: {value}", ln=True)
    pdf.ln(5)

    # Top performer
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Top Performer", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Best-Selling Tour: {top_tour}", ln=True)
    pdf.cell(0, 8, f"Most Popular Day: {top_day}", ln=True)
    pdf.ln(5)

    # Recent bookings table
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Recent Bookings", ln=True)
    pdf.set_font("Arial", "", 11)

    col_widths = [20, 60, 35, 30, 35]
    amount_hdr = "Amount (ex VAT)" if use_ex_vat else "Amount"
    headers = ["#", "Customer", amount_hdr, "Status", "Date"]
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 8, header, border=1)
    pdf.ln()

    def _ex_vat_val(total_val, tax_val):
        try:
            if tax_val is not None:
                return float(total_val) - float(tax_val or 0)
            return float(total_val) / 1.2  # fallback if no tax field
        except Exception:
            return 0.0

    for row in recent_rows:
        total_val = float(row.get("total", 0) or 0)
        tax_val   = row.get("tax_total", None)
        amount_to_show = _ex_vat_val(total_val, tax_val) if use_ex_vat else total_val

        created_dt = row.get("created_date")
        if isinstance(created_dt, (pd.Timestamp, datetime)):
            date_str = created_dt.strftime("%Y-%m-%d")
        else:
            date_str = str(created_dt)[:10]

        pdf.cell(col_widths[0], 8, str(row.get("booking_id", ""))[:10], border=1)
        pdf.cell(col_widths[1], 8, str(row.get("customer_name", ""))[:32], border=1)
        pdf.cell(col_widths[2], 8, f"£{amount_to_show:,.2f}", border=1)
        pdf.cell(col_widths[3], 8, str(row.get("status_name", ""))[:14], border=1)
        pdf.cell(col_widths[4], 8, date_str, border=1)
        pdf.ln()

    return pdf.output(dest="S").encode("latin-1")


# ================================
#  Stock Availability & Missed Revenue (fully guarded)
# ================================
try:
    st.markdown("##  Stock Availability & Missed Revenue")
    st.caption("Filtered to tours only. Ticket totals are the number of tickets sold (EVENT-based).")

    # Use datetime.timedelta and do NOT call .date() — keep as date objects
    stock_start = st.date_input("Start Date for Stock Analysis", value=date.today())
    stock_end   = st.date_input("End Date for Stock Analysis",   value=date.today() + timedelta(days=30))
    num_days = (stock_end - stock_start).days + 1

    # Pull bookings by EVENT DATE (expand items)
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
            n = int(float(v))
            return n if n >= 0 else 0
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

    df_items["event_date"] = df_items["items"].apply(_extract_event_dt_from_items)
    if "date_desc" in df_items.columns:
        df_items["event_date"] = df_items["event_date"].fillna(pd.to_datetime(df_items["date_desc"], errors="coerce"))
    if "created_date" in df_items.columns:
        df_items["event_date"] = df_items["event_date"].fillna(df_items["created_date"])
    df_items["event_date"] = pd.to_datetime(df_items["event_date"], errors="coerce")

    # Filter by EVENT date window
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

    # Capacity & departures
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


    # Stock table
    rows = []
    price_col_label = "Price (ex VAT) (£)"
    lost_col_label  = "Potential Revenue Lost (ex VAT) (£)"

    for product in sorted(df_stock_base["summary"].dropna().unique()):
        pdfp = df_stock_base[df_stock_base["summary"] == product]
        tickets_booked = int(pdfp["ticket_qty"].fillna(0).sum())

        seats_per_dep = TOUR_SEATS_PER_DEP
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

    with st.expander("📋 Full Stock & Revenue Table"):
        st.dataframe(stock_df, use_container_width=True)

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
    # Guard the rest of the app from any stock errors so the PDF button still renders
    with st.expander("⚠️ Stock analysis failed (details)", expanded=False):
        st.exception(e)

# ================================
# PDF build + Sidebar Download
# ================================
def _safe_recent_rows(df):
    if df.empty:
        return [], "N/A", "N/A"
    top_tour = df.groupby("summary")[AMOUNT_COL].sum().idxmax() if "summary" in df.columns else "N/A"
    top_day  = (df["day"].mode()[0] if "day" in df.columns and not df["day"].dropna().empty else "N/A")
    recent   = (
        df.sort_values("created_date", ascending=False)
          .head(5)
          .to_dict(orient="records")
    )
    return recent, top_tour, top_day

recent_rows, top_tour, top_day = _safe_recent_rows(view_booking)
date_range = f"{start.strftime('%d %b %Y')} to {end.strftime('%d %b %Y')}"
logo_path = Path(__file__).parent / "shoebox.png"

pdf_bytes = None
try:
    pdf_bytes = create_detailed_pdf_summary(
        kpi_data=kpi_data,
        date_range=date_range,
        top_tour=top_tour,
        top_day=top_day,
        recent_rows=recent_rows,
        logo_path=str(logo_path) if logo_path.exists() else None,
        use_ex_vat=(AMOUNT_COL == "total_ex_vat"),
    )
except Exception as e:
    with st.sidebar:
        st.warning(f"Could not build PDF: {e}")

with st.sidebar:
    st.markdown("---")
    st.subheader("Report")
    if pdf_bytes:
        st.download_button(
            label="⬇️ Download PDF",
            data=pdf_bytes,
            file_name=f"shoebox_summary_{datetime.today().strftime('%Y-%m-%d')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.button("Download PDF (unavailable)", disabled=True, use_container_width=True)
        st.caption("PDF will appear once there’s data and the report is built.")













