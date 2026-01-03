import io
from pathlib import Path
from datetime import datetime

import pandas as pd
import pydeck as pdk
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from vrp_solver import solve_vrp, routes_to_dataframe
from google_routes import get_route_polyline

# ======================================================
# PDF REPORT GENERATION
# ======================================================

def create_pdf_report(summary, solution_df, logo_path: Path | None = None) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    story = []

    title_text = "<b>G-OPT Route Optimization Report</b><br/><font size=10>Professional VRPTW Summary</font>"
    header_data = []
    if logo_path and logo_path.exists():
        img = Image(str(logo_path), width=2.5*cm, height=2.5*cm, kind='proportional')
        header_data.append([img, Paragraph(title_text, styles['Title'])])
    else:
        header_data.append([Paragraph(title_text, styles['Title'])])

    header_table = Table(header_data, colWidths=[3*cm, 13*cm] if logo_path else [16*cm])
    header_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'CENTER')]))
    story.append(header_table)
    story.append(Spacer(1, 20))

    story.append(Paragraph("Executive Summary", styles['Heading2']))
    summary_data = [
        ["TOTAL ESTIMATED COST", f"€{summary['total_cost']:.2f}", "TOTAL DISTANCE", f"{summary['total_km']:.2f} km"],
        ["FUEL EXPENSE", f"€{summary['fuel_cost']:.2f}", "DRIVER WAGES", f"€{summary['wage_cost']:.2f}"],
        ["UTILIZATION", f"{summary['utilization']:.1f}%", "ACTIVE VEHICLES", f"{summary['active_vehicles']}"],
        ["TOTAL HOURS", f"{summary['total_hrs']:.1f} hrs", "FIXED FEES", f"€{summary['toll_cost']:.2f}"]
    ]
    summary_table = Table(summary_data, colWidths=[4*cm, 4*cm, 4*cm, 4*cm])
    summary_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 25))

    story.append(Paragraph("Route Manifests", styles['Heading2']))
    for vid in sorted(solution_df["vehicle"].unique()):
        vdf = solution_df[solution_df["vehicle"] == vid].sort_values("stop_order")
        if len(vdf) <= 2: continue
        story.append(Spacer(1, 10))
        story.append(Paragraph(f"<b>Vehicle V-{vid}</b>", styles['Heading3']))
        card_data = [["Stop", "Location", "Load", "Time Window", "Service"]]
        for _, row in vdf.iterrows():
            card_data.append([str(row['stop_order']), row['name'][:30], str(row['demand']), f"{row['ready_time']}-{row['due_time']}", f"{row['service_time']}m"])
        ct = Table(card_data, colWidths=[1.5*cm, 7.5*cm, 2*cm, 3.5*cm, 2.5*cm])
        ct.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
        ]))
        story.append(ct)
        story.append(PageBreak())

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ======================================================
# STREAMLIT APP
# ======================================================

st.set_page_config(page_title="G-OPT VRPTW Demo", layout="wide")
logo_path = Path("gopt_logo.png")

# Sidebar
st.sidebar.header("💶 Operating Costs (EUR)")
fuel_price = st.sidebar.number_input("Diesel Price (€/L)", 1.0, 3.0, 1.75)
consumption = st.sidebar.number_input("Fuel Consumption (L/100km)", 5.0, 50.0, 25.0)
driver_wage = st.sidebar.number_input("Driver Wage (€/hr)", 10.0, 80.0, 22.50)
toll_rate = st.sidebar.number_input("Avg. Toll (€/km)", 0.0, 1.0, 0.12)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Solver Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
num_vehicles = st.sidebar.number_input("Number of vehicles", 1, 100, 3)
vehicle_capacity = st.sidebar.number_input("Vehicle capacity", 1, 1000, 7)
use_google = st.sidebar.checkbox("Use Google Maps", value=True)

api_key = None
if use_google:
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        api_key = st.sidebar.text_input("Google API key", type="password")

st.sidebar.markdown("---")
st.sidebar.header("🗺 Map Options")
show_labels = st.sidebar.checkbox("Show customer labels", value=True)
depot_only = st.sidebar.checkbox("Show only depot", value=False)

# Header Logic
header_left, header_right = st.columns([1.2, 4])
with header_left:
    if logo_path.exists(): st.image(str(logo_path), width=110)
with header_right:
    st.markdown("## 🚚 G-OPT Route Optimization (VRPTW + Google Roads)")

st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
st.write("Upload a CSV of your locations and constraints, choose your fleet settings, and G-OPT will compute optimized routes.")

# Divider
st.markdown("""<div style="display: flex; align-items: center; text-align: center; color: #1E3A8A;"><hr style="flex-grow: 1; border: none; border-top: 1px solid #1E3A8A;"><span style="padding: 0 10px;">🚚</span><hr style="flex-grow: 1; border: none; border-top: 1px solid #1E3A8A;"></div>""", unsafe_allow_html=True)

df = pd.read_csv(uploaded_file if uploaded_file else "sample_locations.csv")
st.subheader("📍 Input Locations")
st.info("💡 **Note:** Ensure your CSV respects the exact data format shown below.")
st.dataframe(df, use_container_width=True)

# Original Capacity Check Logic
total_demand = sum(df["demand"].tolist()[1:])
total_capacity = num_vehicles * vehicle_capacity

sum_cols = st.columns(4)
with sum_cols[0]: st.metric("Total demand", total_demand)
with sum_cols[1]: st.metric("Total capacity", total_capacity)
with sum_cols[2]: 
    util = (total_demand / total_capacity * 100) if total_capacity > 0 else 0
    st.metric("Utilization", f"{util:.1f} %")
with sum_cols[3]: st.metric("Vehicles", num_vehicles)

if total_demand > total_capacity:
    st.warning("⚠ Total demand exceeds total system capacity — solver may fail.")

# Optimization
if st.button("🚀 Optimize Routes"):
    with st.spinner("Running VRPTW solver..."):
        # Calling solver without departure_time to avoid TypeError
        routes, total_km, unreachable_idx, total_min = solve_vrp(
            coords=list(zip(df["latitude"], df["longitude"])),
            demands=df["demand"].tolist(),
            vehicle_capacity=vehicle_capacity,
            ready_times=df["ready_time"].tolist(),
            due_times=df["due_time"].tolist(),
            service_times=df["service_time"].tolist(),
            num_vehicles=num_vehicles,
            use_google=use_google,
            api_key=api_key
        )

    if routes is None:
        st.error("❌ No feasible solution found.")
    else:
        # Cost Logic
        fuel_cost = (total_km * (consumption / 100)) * fuel_price
        total_hrs = total_min / 60
        wage_cost = total_hrs * driver_wage
        toll_cost = total_km * toll_rate
        total_op_cost = fuel_cost + wage_cost + toll_cost

        if unreachable_idx:
            bad_names = df.iloc[unreachable_idx]["name"].tolist()
            st.error(f"🚨 **Connectivity Alert:** Unreachable: **{', '.join(bad_names)}**")

        st.success(f"Optimization complete! Total distance = **{total_km:.2f} km**")

        # Display Metrics
        m_cols = st.columns(4)
        m_cols[0].metric("Op. Cost", f"€{total_op_cost:.2f}")
        m_cols[1].metric("Fuel", f"€{fuel_cost:.2f}")
        m_cols[2].metric("Wages", f"€{wage_cost:.2f}")
        m_cols[3].metric("Time", f"{total_hrs:.1f} hrs")

        solution_df = routes_to_dataframe(df, routes)
        st.subheader("🧭 Vehicle Routes")
        active_v = 0
        for vid in sorted(solution_df["vehicle"].unique()):
            vdf = solution_df[solution_df["vehicle"] == vid].sort_values("stop_order")
            load = int(vdf["demand"].sum())
            if len(vdf) > 1 and load > 0:
                active_v += 1
                with st.expander(f"Vehicle {vid} (load {load}/{vehicle_capacity})"):
                    st.write(" → ".join(vdf["name"].tolist()))

        # Downloads
        summary_stats = {
            'total_cost': total_op_cost, 'total_km': total_km, 'fuel_cost': fuel_cost,
            'wage_cost': wage_cost, 'toll_cost': toll_cost, 'total_hrs': total_hrs,
            'utilization': util, 'active_vehicles': active_v
        }
        dl_cols = st.columns(2)
        with dl_cols[0]: st.download_button("⬇ CSV", solution_df.to_csv(index=False).encode("utf-8"), "results.csv")
        with dl_cols[1]:
            pdf_bytes = create_pdf_report(summary_stats, solution_df, logo_path)
            st.download_button("⬇ PDF Report", pdf_bytes, "report.pdf")

        # Map Visualization (Original Full Logic)
        st.subheader("🗺 Route Map")
        is_large = len(df) > 25
        sol_sorted = solution_df.sort_values(["vehicle", "stop_order"]).reset_index(drop=True)
        route_colors = [[255, 99, 71], [30, 144, 255], [34, 139, 34], [255, 165, 0]]
        layers = [pdk.Layer("ScatterplotLayer", data=df.iloc[0:1], get_position="[longitude, latitude]", get_radius=120, get_fill_color=[255, 230, 0])]

        if not depot_only:
            for vid in sol_sorted["vehicle"].unique():
                vdf = sol_sorted[sol_sorted["vehicle"] == vid]
                if len(vdf) <= 1: continue 
                color = route_colors[int(vid) % len(route_colors)]
                ordered_nodes = vdf[["latitude", "longitude"]].values.tolist()
                full_path = []
                for i in range(len(ordered_nodes) - 1):
                    origin, dest = ordered_nodes[i], ordered_nodes[i+1]
                    road_coords = None
                    if not is_large and use_google and api_key:
                        try: road_coords = get_route_polyline(origin, dest, api_key)
                        except: road_coords = None
                    if road_coords:
                        for lat, lon in road_coords: full_path.append([lon, lat])
                    else:
                        full_path.extend([[origin[1], origin[0]], [dest[1], dest[0]]])
                layers.append(pdk.Layer("PathLayer", data=[{"path": full_path}], get_path="path", get_color=color, width_scale=8))
                layers.append(pdk.Layer("ScatterplotLayer", data=vdf, get_position="[longitude, latitude]", get_radius=60, get_fill_color=color))

        st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=df["latitude"].mean(), longitude=df["longitude"].mean(), zoom=10)))
