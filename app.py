import io
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas

from vrp_solver import solve_vrp, routes_to_dataframe
from google_routes import get_route_polyline  # real road paths (polyline)


# ======================================================
# PDF REPORT GENERATION
# ======================================================

def create_pdf_report(total_km, total_demand, total_capacity, num_vehicles, solution_df, logo_path: Path | None = None) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    # --- Header & Logo ---
    header_logo_w, header_logo_h = 3.5 * cm, 3.5 * cm
    padding_top = height - 2.0 * cm
    if logo_path and logo_path.exists():
        c.drawImage(str(logo_path), 2*cm, padding_top - header_logo_h, width=header_logo_w, height=header_logo_h, preserveAspectRatio=True, mask='auto')

    title_x = 2 * cm + header_logo_w + 1 * cm
    c.setFont("Helvetica-Bold", 18)
    c.drawString(title_x, padding_top - 0.5 * cm, "G-OPT Route Optimization Report")
    c.setFont("Helvetica", 11)
    c.drawString(title_x, padding_top - 1.3 * cm, "Professional VRPTW Summary & Detailed Itinerary")

    # --- Summary Metrics ---
    y = padding_top - header_logo_h - 1.5 * cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Summary Metrics")
    y -= 0.8 * cm
    c.setFont("Helvetica", 10)
    utilization = (total_demand / total_capacity * 100) if total_capacity > 0 else 0
    
    c.drawString(2 * cm, y, f"Total Distance: {total_km:.2f} km")
    c.drawString(10 * cm, y, f"Capacity Utilization: {utilization:.1f} %")
    y -= 0.5 * cm
    c.drawString(2 * cm, y, f"Total Demand: {total_demand}")
    c.drawString(10 * cm, y, f"Active Vehicles: {num_vehicles}")
    
    # --- Detailed Routes Table (Restored) ---
    y -= 1.5 * cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Detailed Vehicle Itineraries")
    y -= 0.8 * cm
    
    c.setFont("Helvetica-Bold", 9)
    c.drawString(2*cm, y, "Vehicle")
    c.drawString(4*cm, y, "Stop #")
    c.drawString(6*cm, y, "Location Name")
    c.drawString(12*cm, y, "Demand")
    c.line(2*cm, y-0.2*cm, 19*cm, y-0.2*cm)
    y -= 0.6 * cm

    c.setFont("Helvetica", 9)
    for index, row in solution_df.iterrows():
        # Check if we need a new page
        if y < 3 * cm:
            c.showPage()
            y = height - 3 * cm
            c.setFont("Helvetica", 9)

        c.drawString(2*cm, y, str(row['vehicle']))
        c.drawString(4*cm, y, str(row['stop_order']))
        # Truncate long names to fit
        name = row['name'][:30] + '..' if len(row['name']) > 30 else row['name']
        c.drawString(6*cm, y, name)
        c.drawString(12*cm, y, str(row['demand']))
        y -= 0.5 * cm

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


# ======================================================
# STREAMLIT APP
# ======================================================

st.set_page_config(page_title="G-OPT VRPTW Demo", layout="wide")

logo_path = Path("gopt_logo.png")

# Professional header
header_left, header_right = st.columns([1.2, 4])

with header_left:
    if logo_path.exists():
        st.image(str(logo_path), width=110)

with header_right:
    st.markdown(
        """
        ## 🚚 G-OPT Route Optimization (VRPTW + Google Roads)
        <span style='font-size:16px;'>
        Solve complex <b>Vehicle Routing Problems with Time Windows</b><br>
        using real Google Maps distances & realistic road paths.
        </span>
        """,
        unsafe_allow_html=True,
    )

st.write(
    "Upload a CSV of your locations and constraints, choose your fleet settings, "
    "and G-OPT will compute optimized routes."
)

# ======================================================
# Sidebar Settings
# ======================================================

st.sidebar.header("⚙️ Solver Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file (or use sample_locations.csv)",
    type=["csv"],
)

num_vehicles = st.sidebar.number_input(
    "Number of vehicles",
    min_value=1,
    max_value=100,
    value=3,
)

vehicle_capacity = st.sidebar.number_input(
    "Vehicle capacity (per vehicle)",
    min_value=1,
    max_value=1000,
    value=7,
)

use_google = st.sidebar.checkbox(
    "Use Google Maps (real road distances + paths)",
    value=True,
)

api_key = None
if use_google:
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.sidebar.success("Google API key loaded from secrets ✔")
    else:
        api_key = st.sidebar.text_input(
            "Google API key (local use only)",
            type="password",
            placeholder="AIza...",
        )

st.sidebar.markdown("---")
st.sidebar.header("🗺 Map Options")
show_labels = st.sidebar.checkbox("Show customer labels", value=True)
depot_only = st.sidebar.checkbox("Show only depot (hide routes)", value=False)

# ======================================================
# Load & Display Data
# ======================================================

df = pd.read_csv(uploaded_file if uploaded_file else "sample_locations.csv")

# RESTORED: Input Locations Table
st.subheader("📍 Input Locations")
st.dataframe(df, use_container_width=True)

coords = list(zip(df["latitude"], df["longitude"]))
demands = df["demand"].tolist()
ready_times = df["ready_time"].tolist()
due_times = df["due_time"].tolist()
service_times = df["service_time"].tolist()

# RESTORED: KPI Summary Metrics
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

# RESTORED: Capacity Warning Message
if total_demand > total_capacity:
    st.warning("⚠ Total demand exceeds total system capacity — solver may fail.")

# ======================================================
# Solve VRPTW
# ======================================================

if st.button("🚀 Optimize Routes"):
    with st.spinner("Running VRPTW solver (Calculating real road paths)..."):
        routes, total_km = solve_vrp(
            coords=coords,
            demands=demands,
            vehicle_capacity=vehicle_capacity,
            ready_times=ready_times,
            due_times=due_times,
            service_times=service_times,
            num_vehicles=num_vehicles,
            use_google=use_google,
            api_key=api_key,
        )

    if routes is None:
        st.error("❌ No feasible solution found. Try more vehicles, more capacity, or wider time windows.")
    else:
        st.success(f"Optimization complete! Total distance = **{total_km:.2f} km**")

        solution_df = routes_to_dataframe(df, routes)
        
        # Vehicle Route Expanders
        st.subheader("🧭 Vehicle Routes")
        for vid in sorted(solution_df["vehicle"].unique().tolist()):
            vdf = solution_df[solution_df["vehicle"] == vid].sort_values("stop_order")
            load = int(vdf["demand"].sum())
            with st.expander(f"Vehicle {vid} (load {load}/{vehicle_capacity})"):
                st.write(" → ".join(vdf["name"].tolist()))

        st.subheader("📘 Detailed Route Table")
        st.dataframe(solution_df, use_container_width=True)

        # Downloads
        dl_cols = st.columns(2)
        with dl_cols[0]:
            st.download_button(
                "⬇ Download results as CSV",
                solution_df.to_csv(index=False).encode("utf-8"),
                "vrptw_solution.csv",
                "text/csv",
            )
        with dl_cols[1]:
            pdf_bytes = create_pdf_report(total_km, total_demand, total_capacity, num_vehicles, solution_df, logo_path)
            st.download_button(
                "⬇ Download PDF report",
                pdf_bytes,
                "gopt_report.pdf",
                "application/pdf",
            )

        # ==================================================
        # Map Visualization (Optimized for 100+ Nodes)
        # ==================================================
        st.subheader("🗺 Route Map (Google Roads)")
        
        # Check if dataset is large to decide on drawing style
        # If nodes > 25, we draw straight lines to prevent API timeout and browser crash
        is_large_dataset = len(coords) > 25
        if is_large_dataset:
            st.info("ℹ️ Large dataset detected: Drawing geometric paths for performance. Distance math still uses road data.")

        sol_sorted = solution_df.sort_values(["vehicle", "stop_order"]).reset_index(drop=True)
        route_colors = [[255, 99, 71], [30, 144, 255], [34, 139, 34], [238, 130, 238], [255, 165, 0], [0, 206, 209]]
        
        layers = [
            pdk.Layer(
                "ScatterplotLayer",
                data=df.iloc[0:1],
                get_position="[longitude, latitude]",
                get_radius=120,
                get_fill_color=[255, 230, 0],
                pickable=True,
            )
        ]

        if not depot_only:
            for vid in sol_sorted["vehicle"].unique():
                vdf = sol_sorted[sol_sorted["vehicle"] == vid]
                color = route_colors[int(vid) % len(route_colors)]
                ordered_nodes = vdf[["latitude", "longitude"]].values.tolist()
                full_path = []

                for i in range(len(ordered_nodes) - 1):
                    origin, dest = ordered_nodes[i], ordered_nodes[i+1]
                    
                    road_coords = None
                    # Fetch curvy roads ONLY for small datasets to avoid crashing
                    if not is_large_dataset and use_google and api_key:
                        try:
                            road_coords = get_route_polyline(origin, dest, api_key)
                        except:
                            road_coords = None

                    if road_coords:
                        for lat, lon in road_coords:
                            full_path.append([lon, lat])
                    else:
                        # Fallback to straight lines for speed and stability
                        full_path.extend([[origin[1], origin[0]], [dest[1], dest[0]]])

                layers.append(pdk.Layer(
                    "PathLayer",
                    data=[{"path": full_path}],
                    get_path="path",
                    get_color=color,
                    width_scale=8,
                    width_min_pixels=3,
                ))
                
                layers.append(pdk.Layer(
                    "ScatterplotLayer",
                    data=vdf,
                    get_position="[longitude, latitude]",
                    get_radius=60,
                    get_fill_color=color,
                    pickable=True,
                ))

                if show_labels:
                    layers.append(pdk.Layer(
                        "TextLayer",
                        data=vdf,
                        get_position="[longitude, latitude]",
                        get_text="name",
                        get_size=15,
                        get_color=[0, 0, 0],
                        get_background=True,
                    ))

        view_state = pdk.ViewState(
            latitude=df["latitude"].mean(),
            longitude=df["longitude"].mean(),
            zoom=6 if is_large_dataset else 10
        )

        st.pydeck_chart(pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            map_style=pdk.map_styles.LIGHT,
            tooltip={"text": "{name}"}
        ))
