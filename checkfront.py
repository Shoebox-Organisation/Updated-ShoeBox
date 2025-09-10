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
    s = s.replace("—", "-")         # em-dash → hyphen
    s = s.replace("’", "'")         # curly apostrophe → straight
    s = s.replace("“", '"').replace("”", '"')  # curly quotes → straight
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
st.markdown(
    "<h1 style='text-align: center;'>Shoebox Internal Operations Dashboard</h1>",
    unsafe_allow_html=True
)
st.markdown("---")

# --- Quick Help (top banner) ---
help_box = st.empty()
if not st.session_state.get("applied_once", False):
    help_box.info(
        "First click **Apply filters** in the sidebar on the left. "
        "Then set your filters and press **Apply filters** once more."
    )
else:
    help_box.caption(
        "Tip: adjust filters any time in the sidebar and click **Apply filters**."
    )

# --- Sidebar ---
with st.sidebar:
    # Logo
    logo = Path(__file__).parent / "shoebox.png"
    if logo.exists():
        st.image(str(logo), width=180)

    # Date pickers
    today = date.today()
    start = st.date_input(
        "Start Date", today - timedelta(days=30), key="start_date"
    )
    end = st.date_input(
        "End Date", today + timedelta(days=60), key="end_date"
    )

    # Free-text search
    search = st.text_input(
        "🔍 Search name, email or booking code",
        key="search_text"
    ).strip()

    # Basis choice
    date_basis = st.selectbox(
        "Date basis for KPIs & charts",
        ["Booking date (created)", "Event date"],
        index=0,
        key="date_basis"
    )

    # Categories and products
    cat_to_products = catalog["cat_to_products"]
    all_categories = catalog["all_categories"]

    selected_categories = st.multiselect(
        "Category",
        options=all_categories,
        default=st.session_state.cat_ms,
        key="cat_ms",
        help="Choose one or more categories"
    )

    # Build product pool from chosen categories
    product_pool = sorted({
        p for c in (selected_categories or all_categories)
        for p in cat_to_products.get(c, [])
    })

    # Prune invalid selections
    pruned_items = [p for p in st.session_state.item_ms if p in product_pool]
    if pruned_items != st.session_state.item_ms:
        st.session_state.item_ms = pruned_items

    selected_products = st.multiselect(
        "Items (in selected categories)",
        options=product_pool,
        default=st.session_state.item_ms,
        key="item_ms",
        help="Only items from the selected categories appear here"
    )

    # Placeholders for category/item IDs (if later needed in API calls)
    selected_category_ids: list[str] = []
    selected_item_ids: list[str] = []

    # Status filter placeholder (filled after get_raw runs later)
    status_filter_placeholder = st.empty()

    # Apply / Reset buttons
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

# From here on it's safe to do any data fetching (get_raw, etc.

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
    if include_items:
        common["expand"] = "items"
    if category_ids:
        common["category_id[]"] = category_ids
    if item_ids:
        common["item_id[]"] = item_ids

    # Try server-side windows in this order, else fallback to newest-first:
    def _params_for(field):
        if filter_on == "created":
            return common | {
                f"{field}[min]": start_date.isoformat(),
                f"{field}[max]": end_date.isoformat()
            }
        else:
            return common | {
                "start_date[min]": start_date.isoformat(),
                "start_date[max]": end_date.isoformat()
            }

    param_options = [
        _params_for("created_at"),
        _params_for("created_date"),
        common,
    ]

    def _page_all(params):
        out, seen, page = [], set(), 1
        while True:
            q = params.copy()
            q["page"] = page
            data = _get(q)
            rows = list((data.get("booking/index") or {}).values())
            if not rows:
                break
            for b in rows:
                bid = b.get("booking_id")
                if bid and bid not in seen:
                    seen.add(bid)
                    out.append(b)
            if len(rows) < params["limit"]:
                break
            page += 1
            if max_pages is not None and page > max_pages:
                break
        return out

    # --- FIXED LOOP ---
    rows = []
    for params in param_options:
        try:
            rows = _page_all(params)
            break  # success
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else None
            if code in (400, 403, 500, 502, 503, 504):
                st.caption(f"⚠️ Server rejected params ({code}). Trying fallback…")
                continue  # try next strategy
            raise  # unexpected error
        except requests.RequestException:
            st.caption("⚠️ Network error while fetching; trying fallback…")
            continue

    # Last-resort fallback
    if not rows:
        rows = _page_all(common)

    # Client-side created-date filter as a safety net
    def _created_ts(b):
        for k in ("created_at", "created_date", "created", "date_created", "timestamp_created"):
            v = b.get(k)
            if v is None:
                continue
            ts = pd.to_datetime(v, unit="s", errors="coerce")
            if pd.isna(ts):
                ts = pd.to_datetime(v, errors="coerce")
            if pd.notna(ts):
                try:
                    return ts.tz_localize("UTC").tz_convert(TENANT_TZ).tz_localize(None)
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

def _apply_filters(dfx: pd.DataFrame,
                   selected_categories: list[str],
                   selected_products: list[str],
                   status_filter: str,
                   search: str) -> pd.DataFrame:
    """
    Client-side filters.
    IMPORTANT CHANGE: match categories/products on the WHOLE normalized summary
    (no token splitting) to avoid dropping items like
    'The Matriarchs, Mayors & Merchants Tour'.
    """
    dfx = dfx.copy()

    # Ensure base columns exist
    if "summary" not in dfx.columns:     dfx["summary"] = ""
    if "status_name" not in dfx.columns: dfx["status_name"] = "Unknown"

    # Ensure amount column exists
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

    # ---- Status filter
    if status_filter != "All":
        dfx = dfx[dfx["status_name"].astype(str) == str(status_filter)]

    # ---- Search filter
    s = (search or "").strip().lower()
    if s:
        dfx = dfx[dfx.apply(
            lambda r: s in str(r.get("customer_name", "")).lower()
                   or s in str(r.get("customer_email", "")).lower()
                   or s in str(r.get("code", "")).lower()
                   or s in str(r.get("booking_id", "")).lower(),
            axis=1
        )]

    # ---- Category filter (WHOLE summary match)
    if selected_categories:
        sel_cats = set(selected_categories)

        def _has_selected_cat(summary: str) -> bool:
            # normalize the whole summary and look up its categories
            ns = _norm_title(summary)
            cats = catalog["name_to_cats"].get(ns, set())
            return bool(cats & sel_cats)

        dfx = dfx[dfx["summary"].astype(str).apply(_has_selected_cat)]

    # ---- Product filter (WHOLE summary match)
    if selected_products:
        selected_norm = {_norm_title(p) for p in selected_products}

        def _has_selected_product(summary: str) -> bool:
            return _norm_title(summary) in selected_norm

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

# ---------- DEBUG HELPERS ----------
def _coerce_ts(s):
    s = pd.to_datetime(s, errors="coerce")
    try:
        # drop timezone if present
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_localize(None)
    except Exception:
        pass
    return s

# ---------------------- 🧪 FILTER DIAGNOSTICS ----------------------
with st.expander("🧪 Filter Diagnostics (why nothing shows?)", expanded=False):
    st.write({
        "date_basis": date_basis,
        "start": str(start),
        "end": str(end),
        "status_filter": status_filter,
        "selected_categories": selected_categories,
        "selected_products": selected_products,
        "search": search,
    })

    # Choose the same “base” df you use for rendering
    base_df = (view_event.copy() if date_basis == "Event date" else view_booking.copy())
    basis_col = "event_date" if date_basis == "Event date" else "created_date"

    def _add(step, df, note=""):
        rows = int(len(df))
        dbg_rows.append({"step": step, "rows": rows, "note": note})

    dbg_rows = []

    # 0) Base
    _add("0: Base (before masks)", base_df, f"basis_col={basis_col}")

    # Quick profile
    bts = _coerce_ts(base_df.get(basis_col))
    st.write({
        "basis_min": str(bts.min()) if not base_df.empty else None,
        "basis_max": str(bts.max()) if not base_df.empty else None,
        "status_counts": base_df.get("status_name", pd.Series([], dtype=str)).value_counts().to_dict()
    })

    idx = base_df.index
    M_all = pd.Series(True, index=idx)

    # A) basis ts exists
    M_ts_ok = _coerce_ts(base_df.get(basis_col)).notna()
    _add("A: basis ts notna()", base_df[M_all & M_ts_ok])

    # B) date window
    start_ts = pd.Timestamp(start)
    end_excl = pd.Timestamp(end) + pd.Timedelta(days=1)
    bts = _coerce_ts(base_df.get(basis_col))
    M_range = M_ts_ok & (bts >= start_ts) & (bts < end_excl)
    _add(f"B: in [{start} .. {end}]", base_df[M_all & M_range])

    # C) status
    if status_filter != "All":
        M_status = base_df["status_name"].astype(str) == str(status_filter)
    else:
        M_status = pd.Series(True, index=idx)
    _add("C: status filter", base_df[M_all & M_range & M_status],
         note=f"wanted='{status_filter}'")

    # D) search
    s_txt = (search or "").strip().lower()
    if s_txt:
        M_search = base_df.apply(
            lambda r: s_txt in str(r.get("customer_name","")).lower()
                   or s_txt in str(r.get("customer_email","")).lower()
                   or s_txt in str(r.get("code","")).lower()
                   or s_txt in str(r.get("booking_id","")).lower(),
            axis=1
        )
    else:
        M_search = pd.Series(True, index=idx)
    _add("D: search", base_df[M_all & M_range & M_status & M_search])

    # E) category (mirrors your _apply_filters logic)
    if selected_categories:
        sel_cats = set(selected_categories)
        def _has_cat(summary: str) -> bool:
            for part in _parts(summary):
                if catalog["name_to_cats"].get(part, set()) & sel_cats:
                    return True
            return False
        M_cat = base_df["summary"].astype(str).apply(_has_cat)
    else:
        M_cat = pd.Series(True, index=idx)
    _add("E: category", base_df[M_all & M_range & M_status & M_search & M_cat],
         note=f"selected_categories={selected_categories}")

    # F) products
    if selected_products:
        selp = {_norm_title(p) for p in selected_products}
        def _has_prod(summary: str) -> bool:
            return any(p in selp for p in _parts(summary))
        M_prod = base_df["summary"].astype(str).apply(_has_prod)
    else:
        M_prod = pd.Series(True, index=idx)

    final_mask = M_all & M_range & M_status & M_search & M_cat & M_prod
    _add("F: products", base_df[final_mask], note=f"selected_products={selected_products}")

    st.table(pd.DataFrame(dbg_rows))

    if len(base_df) and final_mask.sum() == 0:
        hints = []
        if M_ts_ok.sum() == 0:
            hints.append(f"No valid timestamps in `{basis_col}` (check date basis or event-date extraction).")
        elif M_range.sum() == 0:
            hints.append("Date window excludes all rows (see basis_min/basis_max above).")
        elif (status_filter != "All") and (M_status.sum() == 0):
            hints.append(f"No rows with status exactly '{status_filter}'.")
        elif selected_categories and (M_cat.sum() == 0):
            hints.append("Category mapping may not match summaries (smart quotes/hyphens).")
        elif selected_products and (M_prod.sum() == 0):
            hints.append("Product names may not match summary parts (normalization).")
        elif s_txt and (M_search.sum() == 0):
            hints.append("Search text matched none of name/email/code/booking_id.")
        st.warning("No rows after filters. Likely causes:\n- " + "\n- ".join(hints))

    # Samples to eyeball
    st.caption("Samples at key steps")
    st.write("**Base**")
    st.dataframe(base_df.head(5))
    st.write("**After date window**")
    st.dataframe(base_df[M_range].head(5))
    st.write("**After status**")
    st.dataframe(base_df[M_range & M_status].head(5))
    st.write("**Final selection**")
    st.dataframe(base_df[final_mask].head(10))
# --------------------------------------------------------------------------------------------


# --- Debug: list unique summaries ---
with st.expander("🔍 Unique summaries in current data"):
    if not current_view.empty:
        unique_summaries = sorted(current_view["summary"].dropna().unique().tolist())
        st.write(f"Found {len(unique_summaries)} unique item summaries")
        st.dataframe(pd.DataFrame(unique_summaries, columns=["summary"]))
    else:
        st.info("No rows in current view.")
        
# --- Debug: show the exact bookings currently used downstream ---
with st.expander("🔎 Bookings pulled from API (current basis + range)", expanded=False):
    basis_col = "event_date" if date_basis == "Event date" else "created_date"

    # Basic stats
    st.write({
        "basis": basis_label,
        "range_inclusive": f"{start} → {end}",
        "rows_in_view": int(len(current_view)),
        "min_basis": str(pd.to_datetime(current_view.get(basis_col), errors="coerce").min()),
        "max_basis": str(pd.to_datetime(current_view.get(basis_col), errors="coerce").max()),
        "status_counts": current_view.get("status_name", pd.Series([], dtype=str)).value_counts().to_dict(),
    })

    # Nicely ordered set of columns for human inspection
    cols = [
        "booking_id", "code",
        basis_col,
        "summary", "status_name",
        "total", "tax_total", "total_ex_vat",
        "paid_total",
        "customer_name", "customer_email",
        "date_desc"  # useful when event-basis
    ]
    cols = [c for c in cols if c in current_view.columns]  # keep only existing
    df_show = (current_view
               .sort_values(basis_col, na_position="last")
               .loc[:, cols])

    st.dataframe(df_show, use_container_width=True, height=420)

    # Quick day/tour rollups to compare with Checkfront totals
    if basis_col in current_view.columns:
        st.caption("Quick rollups to cross-check against Checkfront")
        by_day = (pd.to_datetime(current_view[basis_col], errors="coerce")
                    .dt.date.value_counts().sort_index())
        st.write("Bookings per day:", by_day.to_dict())

        by_item = current_view["summary"].value_counts()
        st.write("Bookings per item:", by_item.to_dict())

    # Export for side-by-side comparison in Excel
    csv = current_view.sort_values(basis_col, na_position="last").to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ Download CSV of current view",
        data=csv,
        file_name=f"shoebox_current_view_{basis_label.lower()}_{start}_{end}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # Optional: inspect one booking's raw JSON (detail endpoint)
    if "booking_id" in current_view.columns:
        bid = st.selectbox(
            "Inspect booking detail (raw JSON)", 
            options=current_view["booking_id"].astype(str).tolist(),
            index=0 if len(current_view) else None
        )
        if st.button("Fetch booking JSON"):
            try:
                detail = fetch_booking_details(bid)
                st.json(detail)
            except Exception as e:
                st.error(f"Detail fetch failed for {bid}: {e}")




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


































