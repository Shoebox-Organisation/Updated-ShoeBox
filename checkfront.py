import streamlit as st
import requests
import pandas as pd
import base64
from datetime import date, timedelta, datetime
from fpdf import FPDF
import plotly.express as px
from pathlib import Path
from dotenv import load_dotenv
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import calendar
import re
import unicodedata
from dotenv import load_dotenv
import streamlit as st
import os# <-- added

# --- PAGE CONFIG (must be first Streamlit call) ---
st.set_page_config(page_title="Shoebox Dashboard", layout="wide")

load_dotenv()

# 🔐 Load environment variables
env_path = Path("C:/Users/PC/Documents/UEA/.env")
load_dotenv(dotenv_path=env_path)

API_KEY = st.secrets.get("API_KEY") or os.getenv("API_KEY")
API_TOKEN = st.secrets.get("API_TOKEN") or os.getenv("API_TOKEN")


if not API_KEY or not API_TOKEN:
    st.error("Missing API credentials. Add API_KEY and API_TOKEN to Streamlit secrets (or .env locally).")
    st.stop()

# --- VAT config ---
VAT_RATE = 0.20  # change here if needed

def ex_vat(amount: float | int | None, rate: float = VAT_RATE) -> float:
    """Fallback ex-VAT calculation when tax_total is unavailable."""
    try:
        return float(amount) / (1.0 + rate)
    except Exception:
        return 0.0

# --- Tour allow-list + normaliser (ADDED) ---
def _norm_title(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).strip().lower()
    s = re.sub(r"\s+", " ", s)      # collapse spaces
    s = s.replace("–", "-")         # en-dash → hyphen
    return s

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

# --- Persistent HTTP session for speed ---
@st.cache_resource
def get_session():
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5,
                  status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry,
                                    pool_connections=20,
                                    pool_maxsize=20))
    return s

SESSION = get_session()

# --- Auth header ---
def get_auth_header():
    token = f"{API_KEY}:{API_TOKEN}"
    b64 = base64.b64encode(token.encode()).decode()
    return {"Authorization": f"Basic {b64}", "Accept": "application/json"}

# --- Fetch bookings (paginated) ---
def fetch_bookings(start_date, end_date, limit=250, include_items=False, max_pages=4000, filter_on="created"):
    base = "https://theshoebox.checkfront.co.uk/api/3.0/booking"
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
        r = SESSION.get(base, headers=headers, params=params, timeout=30)
        r.raise_for_status()
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
    r.raise_for_status()
    return r.json()

# --- Cache API results ---
@st.cache_data(ttl=300)
def get_raw(start, end, include_items=False, filter_on="created"):
    return fetch_bookings(start, end, include_items=include_items, filter_on=filter_on)

# --- Categorisation helper ---
CATEGORY_ORDER = ["Tour", "Group", "Room Hire", "Voucher", "Merchandise", "Fee", "Other"]

# (REPLACED) Strict Tour classification via allow-list
def categorise_product(summary: str) -> str:
    # STRICT: “Tour” only if exact title matches allow-list (normalised)
    if _norm_title(summary) in TOURS_ALLOWLIST:
        return "Tour"

    # Otherwise keep your existing category rules
    s = (summary or "").strip().lower()
    if re.search(r"\broom\b|meeting|hire", s): return "Room Hire"
    if "group" in s: return "Group"
    if "voucher" in s or "gift" in s: return "Voucher"
    if "guidebook" in s or "souvenir" in s or "merch" in s: return "Merchandise"
    if "fee" in s or "reschedul" in s or "cancell" in s or "admin" in s: return "Fee"
    return "Other"

# --- Prepare DataFrame (bookings) ---
@st.cache_data(ttl=300)
def prepare_df(raw):
    df = pd.DataFrame(list(raw.get("booking/index", {}).values()))
    if df.empty:
        return df

    # created_date from any plausible source
    created = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for col in ("created_date", "created", "created_at", "date_created", "timestamp_created"):
        if col in df.columns:
            s = pd.to_datetime(df[col], unit="s", errors="coerce")
            s = s.fillna(pd.to_datetime(df[col], errors="coerce"))
            created = created.fillna(s)
    df["created_date"] = created

    # numerics
    for col in ("total", "paid_total", "tax_total"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # precise net from paid_total if tax_total exists; otherwise fall back to divide-by-rate
    if "tax_total" in df.columns and "paid_total" in df.columns:
        df["total_ex_vat"] = (
            df["paid_total"].fillna(0) - df["tax_total"].fillna(0)
        )
    elif "paid_total" in df.columns:
        df["total_ex_vat"] = df["paid_total"].apply(ex_vat)
    else:
        df["total_ex_vat"] = 0.0

    # labels / helpers
    df["status_name"] = df.get("status_name", "Unknown").fillna("Unknown")
    df["summary"] = df.get("summary", "").astype(str).str.strip()
    df["day"]  = df["created_date"].dt.day_name()
    df["hour"] = df["created_date"].dt.hour
    df["product_category"] = df["summary"].apply(categorise_product)

    # de-dupe
    df = df.drop_duplicates(subset="booking_id", keep="last") if "booking_id" in df.columns else df.drop_duplicates()
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

# --- PDF report function ---
def create_detailed_pdf_summary(kpi_data, date_range, top_tour, top_day, recent_rows, logo_path=None):
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
    headers = ["#", "Customer", "Amount (ex VAT)", "Status", "Date"]
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 8, header, border=1)
    pdf.ln()

    for row in recent_rows:
        # precise ex-VAT per booking when tax_total present
        total_val = row.get("total", 0) or 0
        tax_val = row.get("tax_total", None)
        if tax_val is not None:
            try:
                net = float(total_val) - float(tax_val or 0)
            except Exception:
                net = ex_vat(total_val)
        else:
            net = ex_vat(total_val)

        pdf.cell(col_widths[0], 8, str(row.get("booking_id","")), border=1)
        pdf.cell(col_widths[1], 8, str(row.get("customer_name",""))[:24], border=1)
        pdf.cell(col_widths[2], 8, f"£{net:.2f}", border=1)
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
        elif category_filter == "Tour":  # (CHANGED) strict allow-list + presence in data
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

def _apply_filters(dfx: pd.DataFrame) -> pd.DataFrame:
    if status_filter != "All":
        dfx = dfx[dfx["status_name"] == status_filter]
    if search.strip():
        s = search.strip().lower()
        dfx = dfx[dfx.apply(
            lambda r: s in str(r.get("customer_name","")).lower()
                   or s in str(r.get("customer_email","")).lower()
                   or s in str(r.get("code","")).lower(),
            axis=1
        )]
    if category_filter != "All":
        dfx = dfx[dfx["product_category"] == category_filter]
        # (NEW) enforce allow-list when "Tour" is chosen
        if category_filter == "Tour":
            dfx = dfx[dfx["summary"].astype(str).map(_norm_title).isin(TOURS_ALLOWLIST)]
    if specific_product != "All":
        dfx = dfx[dfx["summary"] == specific_product]
    return dfx

# --- Load booking-date dataset ---
raw_booking = get_raw(start, end, include_items=False, filter_on="created")
df_booking = prepare_df(raw_booking)
df_booking = _apply_filters(df_booking)
cd_booking = pd.to_datetime(df_booking["created_date"], errors="coerce")
view_booking = df_booking[cd_booking.notna() & (cd_booking.dt.date >= start) & (cd_booking.dt.date <= end)].copy()


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

if date_basis == "Event date":
    df_event = get_event_df(start, end)
    df_event = _apply_filters(df_event)
    ed_event = pd.to_datetime(df_event["event_date"], errors="coerce")
    view_event = df_event[ed_event.notna() & (ed_event.dt.date >= start) & (ed_event.dt.date <= end)].copy()

# Choose current view (drives KPIs & charts up to Stock section)
if date_basis == "Event date":
    current_view = view_event.copy()
    current_view["day"] = current_view["event_date"].dt.day_name()
    current_view["hour"] = current_view["event_date"].dt.hour  # usually 0 for all-day events
    basis_series = pd.to_datetime(current_view["event_date"], errors="coerce")
    basis_label = "Event"
else:
    current_view = view_booking.copy()
    basis_series = pd.to_datetime(current_view["created_date"], errors="coerce")
    basis_label = "Booking"

with st.expander("🔎 Debug (selected basis)"):
    st.write("Date basis:", date_basis)
    st.write("Rows (current_view):", len(current_view))
    st.write("Basis min/max:", basis_series.min(), basis_series.max())
    st.write("Category filter:", category_filter)
    st.write("Specific product:", specific_product)

if current_view.empty:
    st.warning("No bookings match this window for the selected date basis and filters.")
    st.stop()

if st.sidebar.button("🔄 Force refresh data"):
    st.cache_data.clear()
    st.cache_resource.clear()

# --- KPIs (BOOKING INDEX ONLY, NET WHEN POSSIBLE) ---
total_bookings = len(current_view)
total_revenue = current_view["total_ex_vat"].sum()
avg_booking = (total_revenue / total_bookings) if total_bookings else 0.0
paid_pct = (current_view["paid_total"] > 0).mean() * 100
repeat_rate = current_view["customer_email"].duplicated().mean() * 100

kpi_data = {
    "Total Bookings": total_bookings,
    "Total Revenue (ex VAT)": f"£{total_revenue:,.2f}",
    "Avg per Booking (ex VAT)": f"£{avg_booking:,.2f}",
    "Paid %": f"{paid_pct:.1f}%",
    "Repeat Customers %": f"{repeat_rate:.1f}%"
}

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Bookings", total_bookings)
k2.metric("Total Revenue (ex VAT)", f"£{total_revenue:,.2f}")
k3.metric("Avg per Booking (ex VAT)", f"£{avg_booking:,.2f}")
k4.metric("Paid %", f"{paid_pct:.1f}%")
k5.metric("Repeat Cust %", f"{repeat_rate:.1f}%")

# --- Charts (BOOKING INDEX ONLY) ---
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
    product_rev = (current_view.groupby("summary")["total_ex_vat"]
                   .sum()
                   .reset_index()
                   .sort_values("total_ex_vat", ascending=False))
    fig3 = px.bar(
        product_rev, x="summary", y="total_ex_vat",
        title=f"💰 Revenue (ex VAT) by Product ({basis_label} Date, Selected Range)", text_auto=True
    )
    fig3.update_yaxes(tickprefix="£", tickformat=",")
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    # === QUARTERLY COMPARISON (APRIL → MARCH) ===
    today = date.today()
    if today.month < 4:
        fy_start = date(today.year - 1, 4, 1)   # Apr 1 last year
        fy_end   = date(today.year, 3, 31)      # Mar 31 this year
    else:
        fy_start = date(today.year, 4, 1)       # Apr 1 this year
        fy_end   = date(today.year + 1, 3, 31)  # Mar 31 next year

    # Get data in the chosen basis
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

    # Build quarter index for the FY window (Q labels will be like 2025Q2 etc.)
    q_index = pd.period_range(start=fy_start, end=fy_end, freq="Q")
    q_series = df_q.groupby("periodQ")["total_ex_vat"].sum()
    q_df = q_series.reindex(q_index, fill_value=0).reset_index()
    q_df.columns = ["quarter", "total"]

    # Hide future quarters within the FY
    q_df.loc[(q_df["quarter"].dt.start_time > pd.Timestamp.today()), "total"] = pd.NA

    # Pretty labels (Q1 = Apr–Jun; Q2 = Jul–Sep; Q3 = Oct–Dec; Q4 = Jan–Mar) and title with FY span
    def _fy_quarter_label(p):
        m = p.start_time.month
        return {4: "Q1 (Apr–Jun)", 7: "Q2 (Jul–Sep)", 10: "Q3 (Oct–Dec)", 1: "Q4 (Jan–Mar)"}.get(m, str(p))

    q_df["quarter"] = q_df["quarter"].apply(_fy_quarter_label)

    fy_label = f"FY {fy_start.year}/{fy_end.year}"

    fig_quarter = px.bar(
        q_df, x="quarter", y="total",
        title=f"Quarterly Revenue (ex VAT) Comparison ({basis_label} Date, {fy_label})",
        text="total"
    )
    fig_quarter.update_traces(texttemplate="£%{y:,.0f}")
    fig_quarter.update_yaxes(tickprefix="£", tickformat=",")
    st.plotly_chart(fig_quarter, use_container_width=True)


# ----------------  MONTHLY (BOOKING INDEX ONLY)  ----------------
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
m_series  = df_m.groupby("periodM")["total_ex_vat"].sum().sort_index()
monthly   = m_series.reindex(month_idx, fill_value=0).reset_index()
monthly.columns = ["Month", "Total Revenue"]
monthly["Month"] = monthly["Month"].dt.strftime("%b %Y")

fig_monthly = px.bar(
    monthly, x="Month", y="Total Revenue",
    title=f"📆 Total Monthly Revenue (ex VAT, {basis_label} Date scope)",
    text="Total Revenue"
)
fig_monthly.update_traces(texttemplate="£%{y:,.0f}")
fig_monthly.update_yaxes(tickprefix="£", tickformat=",")
st.plotly_chart(fig_monthly, use_container_width=True)

# Day/Hour charts
c5, c6 = st.columns(2)

# Day chart (always shown)
with c5:
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    day_counts = current_view["day"].value_counts().reindex(day_order).fillna(0).reset_index()
    day_counts.columns = ["Day", "Count"]
    st.plotly_chart(
        px.bar(day_counts, x="Day", y="Count",
               title=f"📅 Bookings by Day ({basis_label} Date)"),
        use_container_width=True
    )

# Hour chart: ONLY show for Booking date (created)
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
def render_extended_time_comparisons(view_df: pd.DataFrame, basis_series: pd.Series, basis_label: str):
    dfv = view_df.copy()

    # Align a timestamp column to the chosen basis (Booking/Event)
    dfv["_basis_ts"] = pd.to_datetime(basis_series, errors="coerce")
    dfv = dfv[dfv["_basis_ts"].notna()].copy()

    # Week beginning (Monday) as a real date, plus a friendly label for the x-axis
    dfv["week_start"] = dfv["_basis_ts"].dt.to_period("W").apply(lambda r: r.start_time.normalize())
    dfv["week_label"] = dfv["week_start"].dt.strftime("%d %b %Y")
    dfv["year"] = dfv["_basis_ts"].dt.year

    st.markdown("### 📊 Extended Time-Based Comparisons")

    # 1) Weekly Breakdown by Product (ex VAT), x-axis = week beginning date
    st.subheader(f"Weekly Breakdown by Product — {basis_label} date (week beginning)")
    weekly_breakdown = (
        dfv.groupby(["week_start", "week_label", "summary"], as_index=False)["total_ex_vat"]
           .sum()
           .sort_values("week_start")
    )
    fig_wbp = px.bar(
        weekly_breakdown,
        x="week_label", y="total_ex_vat", color="summary", barmode="group",
        title="Weekly Breakdown by Product (ex VAT)"
    )
    fig_wbp.update_yaxes(tickprefix="£", tickformat=",")
    # Keep the labels in chronological order
    ordered_labels = weekly_breakdown.drop_duplicates("week_label")["week_label"].tolist()
    fig_wbp.update_xaxes(categoryorder="array", categoryarray=ordered_labels)
    st.plotly_chart(fig_wbp, use_container_width=True)

    # 2) Weekly Revenue Comparison (ex VAT) — lines by year, x-axis = week beginning date
    st.subheader(f"Weekly Revenue Comparison — {basis_label} date (week beginning)")
    weekly_compare = (
        dfv.groupby(["year", "week_start", "week_label"], as_index=False)["total_ex_vat"]
           .sum()
           .sort_values("week_start")
    )
    fig_wc = px.line(
        weekly_compare,
        x="week_label", y="total_ex_vat", color="year", markers=True,
        title="Weekly Revenue (ex VAT) by Year"
    )
    fig_wc.update_yaxes(tickprefix="£", tickformat=",")
    ordered_labels_wc = weekly_compare.drop_duplicates("week_label")["week_label"].tolist()
    fig_wc.update_xaxes(categoryorder="array", categoryarray=ordered_labels_wc)
    st.plotly_chart(fig_wc, use_container_width=True)

# Call it with your filtered view + basis
render_extended_time_comparisons(current_view, basis_series, basis_label)



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

    # Observed prices (ex VAT) from bookings
    product_prices = {}
    df_tmp = df_stock_base[df_stock_base["ticket_qty"].fillna(0) > 0].copy()
    if not df_tmp.empty:
        df_tmp["unit_price_ex_vat"] = df_tmp.apply(
            lambda r: (ex_vat(r["total"]) / r["ticket_qty"]) if r["ticket_qty"] else 0.0,
            axis=1
        )
        product_prices = df_tmp.groupby("summary")["unit_price_ex_vat"].median().to_dict()

    # ---------- Capacity & departures ----------
    # Business rule: tours run with a max of 12 seats per departure.
    TOUR_SEATS_PER_DEP = 12

    # Tickets per day per product
    perday_tickets = (
        df_stock_base
        .assign(_day=df_stock_base["event_date"].dt.floor("D"))
        .groupby(["summary", "_day"], as_index=False)["ticket_qty"]
        .sum()
    )

    # Estimate departures per day = ceil(tickets_sold_that_day / 12)
    perday_tickets["deps_day"] = (perday_tickets["ticket_qty"] + TOUR_SEATS_PER_DEP - 1) // TOUR_SEATS_PER_DEP

    # Aggregates
    deps_per_day_est = perday_tickets.groupby("summary")["deps_day"].mean().round(2).to_dict()
    days_with_deps   = perday_tickets.groupby("summary")["deps_day"].count().to_dict()
    total_deps       = perday_tickets.groupby("summary")["deps_day"].sum().to_dict()

    # 🔎 Capacity & frequency debug
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

    # --- Build Stock table using fixed 12 seats/dep + AVG departures/day (original chart logic) ---
    rows = []
    cap_sources = {}
    TOUR_SEATS_PER_DEP = 12  # fixed per your rule

    for product in sorted(df_stock_base["summary"].dropna().unique()):
        pdf = df_stock_base[df_stock_base["summary"] == product]
        tickets_booked = int(pdf["ticket_qty"].fillna(0).sum())

        seats_per_dep = TOUR_SEATS_PER_DEP
        cap_sources[product] = "rule-fixed-12"

        # Use the NEW average departures/day, multiplied by the selected #days (original behaviour)
        avg_deps_per_day = float(deps_per_day_est.get(product, 0.0))
        total_capacity = int(round(seats_per_dep * avg_deps_per_day * num_days))

        # Never show negative availability
        available = max(total_capacity - tickets_booked, 0)

        # Observed median unit price (ex VAT)
        avg_price = float(product_prices.get(product, 0.0))
        lost_revenue = available * avg_price

        rows.append({
            "Product": product,
            "Booked Tickets": tickets_booked,
            "Seats/Departure (detected)": seats_per_dep,
            "Avg Departures/Day": round(avg_deps_per_day, 2),
            "Capacity (Seats x Deps x Days)": total_capacity,
            "Available": available,
            "Price ex VAT (£)": round(avg_price, 2),
            "Potential Revenue Lost ex VAT (£)": round(lost_revenue, 2),
        })

    stock_df = pd.DataFrame(rows)

    # --- Debug (keep) ---
    with st.expander("🔎 Debug (stock)"):
        st.write("Capacity source by product:", cap_sources)
        st.write("Products in window:", sorted(df_stock_base["summary"].dropna().unique().tolist()))
        st.write("Rows with non-zero tickets:", int((df_stock_base["ticket_qty"] > 0).sum()))
        st.write("Sample stock rows:", stock_df.head(10))

    # --- Render table + charts (original style) ---
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
        lost = stock_df.sort_values("Potential Revenue Lost ex VAT (£)", ascending=False).copy()
        fig_lost = px.bar(
            lost,
            x="Potential Revenue Lost ex VAT (£)",
            y="Product",
            orientation="h",
            title="💸 Potential Revenue Lost by Product (ex VAT)",
            text="Potential Revenue Lost ex VAT (£)"
        )
        fig_lost.update_yaxes(categoryorder="array", categoryarray=list(lost["Product"]))
        fig_lost.update_xaxes(tickprefix="£", tickformat=",")
        fig_lost.update_traces(
            texttemplate="£%{x:,.0f}",
            hovertemplate="<b>%{y}</b><br>Lost: £%{x:,.2f}<extra></extra>"
        )
        st.plotly_chart(fig_lost, use_container_width=True)


except Exception as e:
    st.warning("⚠️ Error calculating stock and lost revenue.")
    st.error(str(e))




# --- PDF Download (booking-date based view for recent rows/top tour) ---
top_tour = view_booking.groupby("summary")["total_ex_vat"].sum().idxmax() if not view_booking.empty else "N/A"
top_day = view_booking["day"].mode()[0] if not view_booking.empty else "N/A"
recent_rows = view_booking.sort_values("created_date", ascending=False).head(5).to_dict(orient="records")
date_range = f"{start.strftime('%d %b %Y')} to {end.strftime('%d %b %Y')}"
pdf_bytes = create_detailed_pdf_summary(kpi_data, date_range, top_tour, top_day, recent_rows)
today_str = datetime.today().strftime("%Y-%m-%d")
pdf_filename = f"shoebox_summary_{today_str}.pdf"

st.sidebar.download_button(label="⬇️ Download PDF", data=pdf_bytes, file_name=pdf_filename, mime="application/pdf")

