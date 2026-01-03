import io
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor

from vrp_solver import solve_vrp, routes_to_dataframe
from google_routes import get_route_polyline

# ======================================================
# PDF REPORT GENERATION
# ======================================================

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def create_pdf_report(total_km, total_demand, total_capacity, num_vehicles, solution_df, logo_path: Path | None = None) -> bytes:
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
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'CENTER'),
        ('ALIGN', (0,0), (0,0), 'LEFT'),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 20))

    utilization = (total_demand / total_capacity * 100) if total_capacity > 0 else 0
    metric_label = ParagraphStyle('MetricLabel', parent=styles['Normal'], fontSize=8, textColor=colors.grey)
    metric_value = ParagraphStyle('MetricValue', parent=styles['Normal'], fontSize=11, fontName='Helvetica-Bold')

    summary_data = [
        [Paragraph("TOTAL DISTANCE", metric_label), Paragraph("CAPACITY UTILIZATION", metric_label)],
        [Paragraph(f"{total_km:.2f} km", metric_value), Paragraph(f"{utilization:.1f} %", metric_value)],
        [Paragraph("TOTAL DEMAND", metric_label), Paragraph("VEHICLE COUNT", metric_label)],
        [Paragraph(f"{total_demand} units", metric_value), Paragraph(f"{num_vehicles} active", metric_value)]
    ]
    
    summary_table = Table(summary_data, colWidths=[250, 250])
    summary_table.setStyle(TableStyle([
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('BACKGROUND', (0,0), (-1,-1), colors.whitesmoke),
        ('BOX', (0,0), (-1,-1), 1, colors.lightgrey),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 25))

    story.append(Paragraph("Detailed Route Manifest", styles['Heading2']))
    
    data = [["Vehicle", "Stop #", "Location Name", "Demand", "Ready Time"]]
    sorted_df = solution_df.sort_values(['vehicle', 'stop_order'])
    
    for _, row in sorted_df.iterrows():
        data.append([
            f"V-{row['vehicle']}", 
            str(row['stop_order']), 
            row['name'][:35], 
            str(row['demand']),
            str(row['ready_time'])
        ])

    main_table = Table(data, repeatRows=1, colWidths=[60, 50, 240, 60, 80])
    main_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.whitesmoke]),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
    ]))
    
    story.append(main_table)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ======================================================
# STREAMLIT APP
# ======================================================

st.set_page_config(page_title="G-OPT VRPTW Demo", layout="wide")

logo_path = Path("gopt_logo.png")

header_left, header_right = st.columns([1.2, 4])
with header_left:
    if logo_path.exists():
        st.image(str(logo_path), width=110)

with header_right:
    st.markdown("""
        ## 🚚 G-OPT Route Optimization (VRPTW + Google Roads)
        <span style='font-size:16px;'>
        Solve complex <b>Vehicle Routing Problems with Time Windows</b><br>
        using real Google Maps distances & realistic road paths.
        </span>
        """, unsafe_allow_html=True)

st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
st.write("Upload a CSV of your locations and constraints, choose your fleet settings, and G-OPT will compute optimized routes with time windows and capacity constraints.")

# ======================================================
# Sidebar Settings
# ======================================================

st.sidebar.header("⚙️ Solver Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV file (or use sample_locations.csv)", type=["csv"])
num_vehicles = st.sidebar.number_input("Number of vehicles", 1, 100, 3)
vehicle_capacity = st.sidebar.number_input("Vehicle capacity (per vehicle)", 1, 1000, 7)
use_google = st.sidebar.checkbox("Use Google Maps (real road distances + paths)", value=True)

api_key = None
if use_google:
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.sidebar.success("Google API key loaded from secrets ✔")
    else:
        api_key = st.sidebar.text_input("Google API key (local use only)", type="password", placeholder="AIza...")

st.sidebar.markdown("---")
st.sidebar.header("🗺 Map Options")
show_labels = st.sidebar.checkbox("Show customer labels", value=True)
depot_only = st.sidebar.checkbox("Show only depot (hide routes)", value=False)

# Divider
st.markdown("""
    <div style="display: flex; align-items: center; text-align: center; color: #1E3A8A;">
        <hr style="flex-grow: 1; border: none; border-top: 1px solid #1E3A8A;">
        <span style="padding: 0 10px;">🚚</span>
        <hr style="flex-grow: 1; border: none; border-top: 1px solid #1E3A8A;">
    </div>
    """, unsafe_allow_html=True)

df = pd.read_csv(uploaded_file if uploaded_file else "sample_locations.csv")

st.subheader("📍 Input Locations")
st.info("""
    💡 **Note:** The data below is a sample dataset. You can upload your own via the sidebar. 
    **Important:** If uploading your own CSV file, please ensure it **respects the exact data format and column headers** shown in this example to ensure the solver works correctly.
""")

st.dataframe(df, use_container_width=True)

coords = list(zip(df["latitude"], df["longitude"]))
demands = df["demand"].tolist()
ready_times = df["ready_time"].tolist()
due_times = df["due_time"].tolist()
service_times = df["service_time"].tolist()

total_demand = sum(demands[1:])
total_capacity = num_vehicles * vehicle_capacity

sum_cols = st.columns(4)
with sum_cols[0]:
    st.metric("Total demand", total_demand)
with sum_cols[1]:
    st.metric("Total capacity", total_capacity)
with sum_cols[2]:
    util = (total_demand / total_capacity * 100) if total_capacity > 0 else 0
    st.metric("Utilization", f"{util:.1f} %")
with sum_cols[3]:
    st.metric("Vehicles", num_vehicles)

if total_demand > total_capacity:
    st.warning("⚠ Total demand exceeds total system capacity — solver may fail.")

# ======================================================
# Solve VRPTW
# ======================================================

if st.button("🚀 Optimize Routes"):
    with st.spinner("Running VRPTW solver (Calculating real road paths)..."):
        # MODIFIED: Now unpacks 3 variables
        routes, total_km, unreachable_idx = solve_vrp(
            coords=coords, demands=demands, vehicle_capacity=vehicle_capacity,
            ready_times=ready_times, due_times=due_times, service_times=service_times,
            num_vehicles=num_vehicles, use_google=use_google, api_key=api_key
        )

    if routes is None:
        st.error("❌ No feasible solution found. Try more vehicles, more capacity, or wider time windows.")
    else:
        # NEW: Professional Connectivity Alert
        if unreachable_idx:
            bad_names = df.iloc[unreachable_idx]["name"].tolist()
            st.error(f"🚨 **Connectivity Alert:** The following locations are unreachable by road: **{', '.join(bad_names)}**")
            st.warning("Google Maps could not find a driving path to these points (they may be on islands or in restricted zones). A distance penalty has been applied to include them.")

        st.success(f"Optimization complete! Total distance = **{total_km:.2f} km**")

        solution_df = routes_to_dataframe(df, routes)
        
        st.subheader("🧭 Vehicle Routes")
        active_vehicles = 0
        for vid in sorted(solution_df["vehicle"].unique().tolist()):
            vdf = solution_df[solution_df["vehicle"] == vid].sort_values("stop_order")
            load = int(vdf["demand"].sum())
            
            if len(vdf) > 1 and load > 0:
                active_vehicles += 1
                with st.expander(f"Vehicle {vid} (load {load}/{vehicle_capacity})"):
                    st.write(" → ".join(vdf["name"].tolist()))
        
        if active_vehicles == 0:
            st.warning("No cargo was loaded. Check if vehicle capacity is set to 0.")

        st.subheader("📘 Detailed Route Table")
        st.dataframe(solution_df, use_container_width=True)

        dl_cols = st.columns(2)
        with dl_cols[0]:
            st.download_button("⬇ Download results as CSV", solution_df.to_csv(index=False).encode("utf-8"), "vrptw_solution.csv", "text/csv")
        with dl_cols[1]:
            pdf_bytes = create_pdf_report(total_km, total_demand, total_capacity, num_vehicles, solution_df, logo_path)
            st.download_button("⬇ Download PDF report", pdf_bytes, "gopt_report.pdf", "application/pdf")

        # ==================================================
        # Map Visualization
        # ==================================================
        st.subheader("🗺 Route Map (Google Roads)")
        is_large_dataset = len(coords) > 25
        if is_large_dataset:
            st.info("ℹ️ Large dataset detected: Drawing geometric paths for performance.")

        sol_sorted = solution_df.sort_values(["vehicle", "stop_order"]).reset_index(drop=True)
        route_colors = [[255, 99, 71], [30, 144, 255], [34, 139, 34], [238, 130, 238], [255, 165, 0], [0, 206, 209]]
        
        layers = [pdk.Layer("ScatterplotLayer", data=df.iloc[0:1], get_position="[longitude, latitude]", get_radius=120, get_fill_color=[255, 230, 0], pickable=True)]

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
                    if not is_large_dataset and use_google and api_key:
                        try: road_coords = get_route_polyline(origin, dest, api_key)
                        except: road_coords = None

                    if road_coords:
                        for lat, lon in road_coords: full_path.append([lon, lat])
                    else:
                        full_path.extend([[origin[1], origin[0]], [dest[1], dest[0]]])

                layers.append(pdk.Layer("PathLayer", data=[{"path": full_path}], get_path="path", get_color=color, width_scale=8, width_min_pixels=3))
                layers.append(pdk.Layer("ScatterplotLayer", data=vdf, get_position="[longitude, latitude]", get_radius=60, get_fill_color=color, pickable=True))
                if show_labels:
                    layers.append(pdk.Layer("TextLayer", data=vdf, get_position="[longitude, latitude]", get_text="name", get_size=15, get_color=[0, 0, 0], get_background=True))

        st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=df["latitude"].mean(), longitude=df["longitude"].mean(), zoom=6 if is_large_dataset else 10), map_style=pdk.map_styles.LIGHT, tooltip={"text": "{name}"}))
