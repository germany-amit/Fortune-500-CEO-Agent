# Fortune 500 Foresight Digital Twin with Multi-Agent Simulation + PDF Export
# Free-tier friendly (Streamlit Cloud + GitHub)

import streamlit as st
import pandas as pd
import numpy as np
import requests
import math
from datetime import datetime, timezone, timedelta
import folium
from streamlit_folium import st_folium

# PDF
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

st.set_page_config(page_title="Fortune 500 Foresight AI", page_icon="🛰️", layout="wide")

# ------------------------- Styles -------------------------
st.markdown("""
<style>
.block-container {padding-top:.6rem;}
.panel{border:1px solid #e5e7eb;border-radius:14px;padding:12px;background:#fff;box-shadow:0 6px 18px rgba(0,0,0,.05)}
.kpi{font-size:.9rem;color:#64748b}
.badge{display:inline-block;padding:.2rem .6rem;border-radius:9999px;background:#eef2ff;color:#3730a3;font-weight:600;margin-right:.5rem}
.thinking{font-variant-caps:all-small-caps;letter-spacing:.06em;opacity:.95}
table td, table th {vertical-align: top !important;}
</style>
""", unsafe_allow_html=True)

# ------------------------- Navigation -------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Dashboard", "Agents (Debrief & Consensus)", "5W+1H Use Case"])

# ------------------------- Demo Sites -------------------------
def demo_sites():
    return pd.DataFrame([
        {"name":"Delhi HQ",      "lat":28.6139, "lon":77.2090},
        {"name":"NY DataCenter", "lat":40.7128, "lon":-74.0060},
        {"name":"Tokyo Plant",   "lat":35.6762, "lon":139.6503},
        {"name":"Berlin Office", "lat":52.5200, "lon":13.4050},
        {"name":"Singapore Hub", "lat":1.3521,  "lon":103.8198},
        {"name":"São Paulo DC",  "lat":-23.5505,"lon":-46.6333}
    ])

# ------------------------- Sidebar Controls -------------------------
with st.sidebar:
    st.subheader("📍 Sites")
    up = st.file_uploader("Upload CSV (name,lat,lon)", type=["csv"])
    st.caption("If omitted, demo sites load.")

    st.subheader("📡 IoT Sensor APIs")
    use_quakes  = st.checkbox("USGS Earthquakes (24h)", True)
    use_weather = st.checkbox("Open-Meteo Weather (site)", True)
    use_aq      = st.checkbox("OpenAQ PM2.5 (site)", True)
    use_cisa    = st.checkbox("CISA KEV (30d)", True)
    use_gdelt   = st.checkbox("GDELT Unrest (24h)", True)

    st.subheader("⚖️ Edge Risk Weights")
    w_quake  = st.slider("Seismic", 0.0, 1.0, 0.30, 0.05)
    w_weather= st.slider("Weather", 0.0, 1.0, 0.25, 0.05)
    w_air    = st.slider("Air Quality", 0.0, 1.0, 0.15, 0.05)
    w_cyber  = st.slider("Cyber", 0.0, 1.0, 0.15, 0.05)
    w_unrest = st.slider("Unrest", 0.0, 1.0, 0.15, 0.05)
    st.caption("Weights auto-normalized.")

    st.subheader("🔁 Refresh")
    st.button("Refresh Now")  # manual refresh (free tier)

# ------------------------- Sites Load -------------------------
if up:
    try:
        sites = pd.read_csv(up)
        assert set(["name","lat","lon"]).issubset(sites.columns)
        sites = sites[["name","lat","lon"]].dropna()
    except Exception as e:
        st.error(f"CSV error: {e}")
        sites = demo_sites()
else:
    sites = demo_sites()

if sites.empty:
    st.stop()

# ------------------------- Utils -------------------------
def normalize_weights(lst):
    s = sum(lst); 
    return [x/s if s>0 else 0 for x in lst]

def haversine_km(a_lat, a_lon, b_lat, b_lon):
    R=6371.0
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dphi = math.radians(b_lat-a_lat)
    dl   = math.radians(b_lon-a_lon)
    x = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(x))

@st.cache_data(ttl=300)
def safe_get_json(url, timeout=12):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

wq, ww, wa, wc, wu = normalize_weights([w_quake, w_weather, w_air, w_cyber, w_unrest])

# ------------------------- IoT: Sensor APIs -------------------------
@st.cache_data(ttl=300)
def iot_quakes_24h():
    if not use_quakes: return []
    try:
        j = safe_get_json("https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson")
        rows=[]
        for f in j.get("features", []):
            c = f.get("geometry",{}).get("coordinates",[None,None])
            if c and c[0] is not None and c[1] is not None:
                rows.append({
                    "lat":c[1], "lon":c[0],
                    "mag": float(f.get("properties",{}).get("mag") or 0),
                    "place": f.get("properties",{}).get("place","")
                })
        return rows
    except Exception:
        return []

@st.cache_data(ttl=300)
def iot_weather(lat, lon):
    if not use_weather: return None
    try:
        j = safe_get_json(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true")
        cw = j.get("current_weather", {})
        return {"temp_c": cw.get("temperature"), "windspeed": cw.get("windspeed")}
    except Exception:
        return None

@st.cache_data(ttl=600)
def iot_air_pm25(lat, lon):
    if not use_aq: return None
    try:
        j = safe_get_json(f"https://api.openaq.org/v2/latest?coordinates={lat},{lon}&radius=50000&parameter=pm25&limit=1&order_by=measurements_value&sort=desc")
        res = j.get("results", [])
        if not res: return None
        meas = res[0].get("measurements", [])
        if not meas: return None
        return float(meas[0].get("value"))
    except Exception:
        return None

@st.cache_data(ttl=3600)
def iot_cisa_kev_count(days=30):
    if not use_cisa: return 0
    try:
        j = safe_get_json("https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json")
        items = j.get("vulnerabilities", [])
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        cnt=0
        for it in items:
            d=it.get("dateAdded")
            if not d: continue
            try:
                dt=datetime.fromisoformat(d.replace("Z","+00:00"))
                if dt>=cutoff: cnt+=1
            except Exception:
                continue
        return cnt
    except Exception:
        return 0

@st.cache_data(ttl=900)
def iot_unrest_mentions(hours=24):
    if not use_gdelt: return 0
    try:
        q='("protest" OR "strike" OR "unrest" OR "riot")'
        url=f"https://api.gdeltproject.org/api/v2/doc/doc?query={requests.utils.quote(q)}&format=json&mode=artlist&maxrecords=100&timespan={hours}h"
        j=safe_get_json(url)
        return len(j.get("articles", []))
    except Exception:
        return 0

# Fetch shared feeds once
quakes = iot_quakes_24h()
kev_recent = iot_cisa_kev_count(30)
unrest_24h = iot_unrest_mentions(24)

# ------------------------- Edge: Scoring -------------------------
def edge_seismic_score(lat, lon):
    if not quakes: return 0.0, None
    best=0.0; note=None
    for q in quakes:
        d = haversine_km(lat, lon, q["lat"], q["lon"])
        if d<=500 and q["mag"]>best:
            best=q["mag"]; note=f"M{q['mag']:.1f} near {q['place']}"
    return min(best/8.0,1.0), note

def edge_weather_score(w):
    if not w: return 0.0, None
    s=0.0; notes=[]
    t=w.get("temp_c"); ws=w.get("windspeed")
    if t is not None:
        if t>=40: s=max(s,0.8); notes.append(f"heat {t:.0f}°C")
        elif t>=32: s=max(s,0.5); notes.append(f"warm {t:.0f}°C")
        elif t<=-5: s=max(s,0.6); notes.append(f"freeze {t:.0f}°C")
    if ws is not None:
        if ws>=70: s=max(s,0.8); notes.append(f"wind {ws:.0f} km/h")
        elif ws>=40: s=max(s,0.5); notes.append(f"wind {ws:.0f} km/h")
    return min(s,1.0), (", ".join(notes) if notes else None)

def edge_air_score(pm25):
    if pm25 is None: return 0.0, None
    if pm25<=50:     return 0.1, f"PM2.5 {pm25:.0f} (Good)"
    if pm25<=100:    return 0.3, f"PM2.5 {pm25:.0f} (Moderate)"
    if pm25<=150:    return 0.5, f"PM2.5 {pm25:.0f} (Unhealthy SG)"
    if pm25<=250:    return 0.7, f"PM2.5 {pm25:.0f} (Unhealthy)"
    return 0.9, f"PM2.5 {pm25:.0f} (Very Unhealthy)"

def edge_cyber_score(cnt):
    if cnt>=50:  return 0.8, f"KEV +{cnt}/30d"
    if cnt>=25:  return 0.6, f"KEV +{cnt}/30d"
    if cnt>=10:  return 0.4, f"KEV +{cnt}/30d"
    if cnt>=1:   return 0.2, f"KEV +{cnt}/30d"
    return 0.0, "KEV stable"

def edge_unrest_score(cnt):
    if cnt>=60:  return 0.8, f"Unrest {cnt}/24h"
    if cnt>=30:  return 0.6, f"Unrest {cnt}/24h"
    if cnt>=10:  return 0.4, f"Unrest {cnt}/24h"
    if cnt>=1:   return 0.2, f"Unrest {cnt}/24h"
    return 0.0, "Unrest low"

def risk_to_color(x):
    t=float(max(0.0, min(1.0, x)))
    if t<0.33:   return "green"
    elif t<0.66: return "orange"
    else:        return "red"

def robot_playbook(r):
    acts=[]
    if r["seismic"]>=0.6: acts.append("Seismic checks; pause critical ops; drone inspection")
    elif r["seismic"]>=0.3: acts.append("Post-tremor inspection")
    if r["weather"]>=0.6: acts.append("Severe weather posture; secure assets; power backup")
    elif r["weather"]>=0.3: acts.append("Heat/cold plan; hydration/PPE")
    if r["air"]>=0.7: acts.append("N95; indoor shift; filtration")
    elif r["air"]>=0.4: acts.append("Improve ventilation; monitor AQ")
    if r["cyber"]>=0.6: acts.append("Patch now; MFA; network segmentation")
    elif r["cyber"]>=0.3: acts.append("Heighten SOC monitoring")
    if r["unrest"]>=0.6: acts.append("Reroute logistics; travel freeze")
    elif r["unrest"]>=0.3: acts.append("Staff advisories; security briefing")
    return " | ".join(acts) if acts else "Normal posture — continue operations"

# Build enriched site table (used in multiple pages)
def build_enriched(sites_df):
    rows=[]
    for _, s in sites_df.iterrows():
        lat, lon = float(s["lat"]), float(s["lon"])

        w = iot_weather(lat, lon) if use_weather else None
        pm= iot_air_pm25(lat, lon) if use_aq else None
        qs, qnote = edge_seismic_score(lat, lon) if use_quakes else (0.0, None)
        ws, wnote = edge_weather_score(w) if use_weather else (0.0, None)
        ascore, anote = edge_air_score(pm) if use_aq else (0.0, None)
        cs, cnote = edge_cyber_score(kev_recent) if use_cisa else (0.0, None)
        us, unote = edge_unrest_score(unrest_24h) if use_gdelt else (0.0, None)

        total = wq*qs + ww*ws + wa*ascore + wc*cs + wu*us

        rows.append({
            "name": s["name"], "lat": lat, "lon": lon,
            "risk": round(total,3),
            "seismic": qs, "seismic_note": qnote,
            "weather": ws, "weather_note": wnote,
            "air": ascore, "air_note": anote,
            "cyber": cs, "cyber_note": cnote,
            "unrest": us, "unrest_note": unote,
            "temp_c": None if not w else w.get("temp_c"),
            "windspeed": None if not w else w.get("windspeed"),
            "pm25": pm
        })
    return pd.DataFrame(rows)

enriched = build_enriched(sites)

# ------------------------- Shared Headers -------------------------
def thinking_header():
    c1,c2,c3 = st.columns(3)
    with c1: st.info(f"📡 IoT — sensing…  | Quakes {len(quakes) if use_quakes else 0} | KEV {kev_recent if use_cisa else 0} | Unrest {unrest_24h if use_gdelt else 0}")
    with c2: st.warning("⚡ Edge — anomaly detection & risk scoring…")
    with c3: st.success("🤖 Robot — site playbooks ready…")

# ------------------------- PDF Export -------------------------
def generate_executive_pdf(enriched_df, kev_recent, unrest_24h):
    """
    Build a 1–2 page executive brief (5W+1H, KPIs, Top Sites + Robot Actions).
    """
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    left, top = 2*cm, height - 2*cm
    line = top

    def h(text, size=16, gap=10):
        nonlocal line
        c.setFont("Helvetica-Bold", size)
        c.drawString(left, line, text)
        line -= gap

    def p(text, size=10, wrap=88, gap=12):
        # simple word wrap
        nonlocal line
        c.setFont("Helvetica", size)
        words = text.split()
        current = ""
        for w in words:
            if len(current + " " + w) > wrap:
                c.drawString(left, line, current)
                line -= gap
                current = w
            else:
                current = w if current == "" else current + " " + w
        if current:
            c.drawString(left, line, current)
            line -= gap

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    # Cover / Header
    h("Fortune 500 Foresight Executive Brief", 18, 16)
    p(f"Generated: {now}")
    line -= 6

    # KPIs
    high = int((enriched_df['risk']>=0.66).sum())
    med  = int(((enriched_df['risk']>=0.33)&(enriched_df['risk']<0.66)).sum())
    low  = int((enriched_df['risk']<0.33).sum())
    p(f"Sites: {len(enriched_df)}  |  High Risk: {high}  |  Medium Risk: {med}  |  Low Risk: {low}")
    p(f"CISA KEV (30d): {kev_recent}  |  Unrest Mentions (24h): {unrest_24h}")
    line -= 6

    # 5W+1H
    h("5W + 1H", 14, 14)
    p("What: Planet Digital Twin combining IoT sensors (via APIs), Edge analytics, and AI/Robotics for real-time resilience intelligence.")
    p("Why: Fortune 500 face climate risk, supply chain shocks, energy crises, and global uncertainty; traditional systems are slow, siloed, or expensive.")
    p("Who: CxOs (CEO/CTO/CSO/CDO), Risk & Resilience leaders, Supply Chain/Operations heads; IoT/Edge/AI teams.")
    p("Where: Global HQs, smart factories, logistics hubs, data centers, financial control rooms.")
    p("When: Real-time monitoring now; toward autonomous IoT-Edge-AI agents by 2050–2100.")
    p("How: APIs as sensors → Edge anomaly detection + weighted risk scoring → Robot playbooks → Agent debriefs → Executive map/KPIs/CSV.")

    # Top sites table
    line -= 6
    h("Top Sites & Robot Actions", 14, 14)
    top5 = enriched_df.sort_values("risk", ascending=False).iloc[:5]
    c.setFont("Helvetica-Bold", 10)
    c.drawString(left, line, "Site")
    c.drawString(left+7*cm, line, "Risk")
    c.drawString(left+10*cm, line, "Action")
    line -= 12
    c.setFont("Helvetica", 10)
    for _, r in top5.iterrows():
        action = robot_playbook(r)
        c.drawString(left, line, str(r['name'])[:28])
        c.drawString(left+7*cm, line, f"{r['risk']:.2f}")
        # Wrap action text
        action_lines = []
        a_words = action.split()
        cur=""
        for w in a_words:
            if len(cur + " " + w) > 48:
                action_lines.append(cur)
                cur = w
            else:
                cur = w if not cur else cur + " " + w
        if cur:
            action_lines.append(cur)
        for i, al in enumerate(action_lines):
            c.drawString(left+10*cm, line, al)
            if i < len(action_lines)-1:
                line -= 12
        line -= 14
        if line < 3*cm:
            c.showPage(); line = top

    # End
    c.showPage()
    c.save()
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes

# =========================
# Page 1: Dashboard
# =========================
if page == "Dashboard":
    st.title("🌍 Fortune 500 Foresight Intelligence Dashboard")
    st.caption("IoT sensors via APIs → Edge analytics → Robot actions. Free Streamlit + GitHub.")

    thinking_header()

    # Map
    st.subheader("🗺️ Live Risk Map")
    m = folium.Map(location=[20,0], zoom_start=2, tiles="CartoDB positron")
    for _, r in enriched.iterrows():
        color = risk_to_color(r["risk"])
        popup = (f"<b>{r['name']}</b><br>"
                 f"Risk: {r['risk']:.2f}<br>"
                 f"{r['seismic_note'] or ''}<br>"
                 f"{r['weather_note'] or ''}<br>"
                 f"{r['air_note'] or ''}<br>"
                 f"{r['cyber_note'] or ''}<br>"
                 f"{r['unrest_note'] or ''}<br>")
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=9, color=color, fill=True, fill_opacity=0.85,
            popup=folium.Popup(popup, max_width=280)
        ).add_to(m)
    if use_quakes and quakes:
        for q in quakes[:50]:
            folium.CircleMarker(
                location=[q["lat"], q["lon"]],
                radius=4, color="red", fill=True, fill_opacity=0.6,
                tooltip=f"M{q['mag']:.1f} {q['place']}"
            ).add_to(m)
    st_folium(m, width=1000, height=420)

    # KPIs
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Sites", len(enriched))
    k2.metric("High Risk (≥0.66)", int((enriched["risk"]>=0.66).sum()))
    k3.metric("Med Risk (0.33–0.66)", int(((enriched["risk"]>=0.33)&(enriched["risk"]<0.66)).sum()))
    k4.metric("CISA KEV (30d)", kev_recent if use_cisa else 0)
    k5.metric("Unrest Mentions (24h)", unrest_24h if use_gdelt else 0)

    # Table + Robot actions
    st.subheader("🧭 Site Risk & Robot Actions")
    view = enriched.copy()
    view["IoT (Weather)"] = view.apply(lambda r: f"{'' if r['temp_c'] is None else str(round(r['temp_c']))+'°C'}, {'' if r['windspeed'] is None else str(round(r['windspeed']))+' km/h'}", axis=1)
    view["IoT (Air)"] = view.apply(lambda r: f"{'' if r['pm25'] is None else str(int(r['pm25']))} μg/m³", axis=1)
    view["Robot Action"] = view.apply(robot_playbook, axis=1)
    st.dataframe(view[["name","risk","seismic_note","IoT (Weather)","air_note","cyber_note","unrest_note","Robot Action"]].sort_values("risk", ascending=False), use_container_width=True)

    # Export CSV
    csv = view[["name","lat","lon","risk","seismic_note","weather_note","air_note","cyber_note","unrest_note","Robot Action"]].sort_values("risk", ascending=False).to_csv(index=False)
    st.download_button("⬇️ Download CSV Report", csv, file_name="foresight_risk_report.csv", mime="text/csv")

    # Export PDF
    st.subheader("🧾 Executive Brief (PDF)")
    if st.button("Generate Executive Brief PDF"):
        pdf_bytes = generate_executive_pdf(enriched, kev_recent, unrest_24h)
        st.download_button(
            "Download Executive Brief PDF",
            data=pdf_bytes,
            file_name="Executive_Brief_Foresight.pdf",
            mime="application/pdf"
        )

# =========================
# Page 2: Agents (Debrief & Consensus)
# =========================
elif page == "Agents (Debrief & Consensus)":
    st.title("🤝 Multi-Agent Debrief & Consensus")
    st.caption("Rule-based lightweight agents (free tier) brief from their perspective, then form a consensus recommendation.")

    thinking_header()

    # Aggregate signals for agents
    high_risk_sites = enriched[enriched["risk"]>=0.66].sort_values("risk", ascending=False)["name"].tolist()
    med_risk_sites  = enriched[(enriched["risk"]>=0.33)&(enriched["risk"]<0.66)].sort_values("risk", ascending=False)["name"].tolist()
    top_site = enriched.sort_values("risk", ascending=False).iloc[0] if len(enriched) else None

    def j(lst, n=5):
        return ", ".join(lst[:n]) if lst else "None"

    with st.expander("🛡️ CISO Agent (Cyber)"):
        if use_cisa:
            if kev_recent>=50:
                st.write(f"- Elevated KEV volume (**{kev_recent}/30d**). Recommend **patch sprint + MFA** across all sites.")
            elif kev_recent>=25:
                st.write(f"- Moderate KEV volume (**{kev_recent}/30d**). Recommend **accelerated patching** & **SOC heightened monitoring**.")
            elif kev_recent>=10:
                st.write(f"- Noticeable KEV volume (**{kev_recent}/30d**). Maintain patch cadence; validate backups.")
            else:
                st.write(f"- Low KEV volume (**{kev_recent}/30d**). Regular hygiene sufficient.")
        else:
            st.write("- Cyber feed disabled.")

    with st.expander("🛰️ Risk Agent (Seismic/Weather/Air/Unrest)"):
        if top_site is not None:
            st.write(f"- Highest risk site: **{top_site['name']}** (score **{top_site['risk']:.2f}**).")
            if top_site["seismic_note"]: st.write(f"  • Seismic: {top_site['seismic_note']}")
            if top_site["weather_note"]: st.write(f"  • Weather: {top_site['weather_note']}")
            if top_site["air_note"]:     st.write(f"  • Air: {top_site['air_note']}")
            if top_site["unrest_note"]:  st.write(f"  • Unrest: {top_site['unrest_note']}")
        st.write(f"- High-risk sites (≥0.66): {j(high_risk_sites)}")
        st.write(f"- Medium-risk sites (0.33–0.66): {j(med_risk_sites)}")

    with st.expander("🚚 Supply Chain Agent"):
        if len(high_risk_sites)>0:
            st.write(f"- **Reroute logistics** around: {j(high_risk_sites)}")
            st.write("- Pre-position inventory and diversify carriers.")
        elif len(med_risk_sites)>0:
            st.write(f"- **Monitor and plan alternates** for: {j(med_risk_sites)}")
        else:
            st.write("- No reroutes required. Maintain standard routes.")

    with st.expander("💰 CFO Agent"):
        if len(high_risk_sites)>0:
            st.write("- Approve **contingency spend** for resilience at high-risk sites (backup power, spares, PPE).")
        st.write("- Prioritize **patching budget** aligned to KEV levels.")
        st.write("- Optimize insurance coverage for regions with persistent risk.")

    st.markdown("---")
    def consensus(enriched_df):
        recs=[]
        if (enriched_df["risk"]>=0.66).any():
            recs.append("🔴 **Activate regional resilience posture** for high-risk sites (Ops + Security + Facilities).")
        if kev_recent>=25 and use_cisa:
            recs.append("🟠 **Run 7-day patch sprint + enforce MFA** across critical systems.")
        if unrest_24h>=30 and use_gdelt:
            recs.append("🟠 **Issue travel & site access advisories** in impacted regions.")
        if (enriched_df["air"]>=0.7).any():
            recs.append("🟠 **Deploy N95 & indoor shift** at poor air quality sites.")
        if not recs:
            recs.append("🟢 **Steady-state operations** — continue monitoring.")
        return recs

    st.subheader("🧭 Consensus Recommendation")
    for r in consensus(enriched):
        st.write("- " + r)

    st.subheader("📋 Top 3 Sites — Action Brief")
    if len(enriched):
        top3 = enriched.sort_values("risk", ascending=False).iloc[:3]
        for _, row in top3.iterrows():
            st.markdown(f"**{row['name']}** (risk {row['risk']:.2f})  \n"
                        f"- {row['seismic_note'] or 'No seismic note'}  \n"
                        f"- {row['weather_note'] or 'No weather note'}  \n"
                        f"- {row['air_note'] or 'No air note'}  \n"
                        f"- {row['cyber_note'] or 'No cyber note'}  \n"
                        f"- {row['unrest_note'] or 'No unrest note'}")

# =========================
# Page 3: 5W+1H
# =========================
else:
    st.title("📌 5W + 1H — Executive Use Case (Fortune 500)")
    st.markdown("""
### **What**  
A **Planet Digital Twin** combining **IoT sensor data (via APIs), Edge analytics, and AI/Robotics** to deliver **real-time resilience intelligence**.

### **Why**  
Fortune 500 companies face **climate risk, supply chain disruptions, energy crises, and global uncertainties**. Traditional systems are **slow, siloed, or expensive**.  
This app shows how **2050–2100 AI + IoT integration** can be done **today** using **open-source tools**.

### **Who**  
- CxOs (CEO/CTO/CSO/CDO), Risk & Resilience leaders, Supply Chain/Operations heads  
- IoT/Edge/AI innovation teams, Finance/Insurance/ESG stakeholders

### **Where**  
Global HQs, smart factories, logistics hubs, data centers, and financial control rooms.

### **When**  
- **Now**: Real-time monitoring & alerts  
- **Future** (2050–2100): Autonomous IoT-Edge-AI agents orchestrating resilient operations

### **How**  
- **IoT**: Public APIs as proxy for global hardware sensors (seismic, weather, air, cyber, unrest)  
- **Edge**: On-the-fly anomaly detection, **weighted risk scoring** with explainable notes  
- **Robot**: Site-specific operational playbooks  
- **Agents**: Role-based debriefs (Risk, CISO, Supply Chain, CFO) + **consensus**  
- **Viz**: Executive map & KPIs; CSV/PDF export for audit
""")
    st.success("This section is designed so decision-makers instantly see the business value and governance story.")
