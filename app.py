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
# PDF REPORT GENERATION (clean header, schematic map)
# ======================================================

def create_pdf_report(
    total_km,
    total_demand,
    total_capacity,
    num_vehicles,
    solution_df,
    logo_path: Path | None = None,
) -> bytes:
    """Create a professional PDF report with logo, KPIs, schematic map & vehicle routes."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # ------------------------------------------------------------------
    # CLEAN PROFESSIONAL HEADER (logo left, text right)
    # ------------------------------------------------------------------
    header_logo_w = 3.5 * cm
    header_logo_h = 3.5 * cm
    padding_top = height - 2.0 * cm

    # Logo on the left
    if logo_path and logo_path.exists():
        try:
            c.drawImage(
                str(logo_path),
                2 * cm,                          # X position
                padding_top - header_logo_h,     # Y position
                width=header_logo_w,
                height=header_logo_h,
                preserveAspectRatio=True,
                mask="auto",
            )
        except Exception:
            pass

    # Title & subtitle to the right of logo
    title_x = 2 * cm + header_logo_w + 1 * cm
    title_y = padding_top - 0.5 * cm

    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(HexColor("#000000"))
    c.drawString(title_x, title_y, "G-OPT Route Optimization Report")

    c.setFont("Helvetica", 11)
    c.drawString(title_x, title_y - 0.8 * cm, "Professional VRPTW Summary Report")

    # Start writing content below header
    y = padding_top - header_logo_h - 1.5 * cm

    # ------------------------------------------------------------------
    # KPI SUMMARY (2-column layout)
    # ------------------------------------------------------------------
    c.setFillColor(HexColor("#000000"))
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Summary")
    y -= 0.7 * cm

    c.setFont("Helvetica", 10)

    utilization = (total_demand / total_capacity * 100) if total_capacity > 0 else 0

    vehicles = sorted(solution_df["vehicle"].unique().tolist())
    avg_stops = 0.0
    if len(vehicles) > 0:
        num_non_depot_stops = solution_df[solution_df["name"] != "Depot"].shape[0]
        avg_stops = num_non_depot_stops / len(vehicles)

    left_x = 2 * cm
    right_x = width / 2 + 0.5 * cm

    y_left = y
    c.drawString(left_x, y_left, f"Total distance: {total_km:.2f} km")
    y_left -= 0.5 * cm
    c.drawString(left_x, y_left, f"Total demand: {total_demand}")
    y_left -= 0.5 * cm
    c.drawString(left_x, y_left, f"Total capacity: {total_capacity}")

    y_right = y
    c.drawString(right_x, y_right, f"Load utilization: {utilization:.1f} %")
    y_right -= 0.5 * cm
    c.drawString(right_x, y_right, f"Number of vehicles: {num_vehicles}")
    y_right -= 0.5 * cm
    c.drawString(right_x, y_right, f"Avg stops / vehicle: {avg_stops:.1f}")

    y = min(y_left, y_right) - 1.0 * cm

    # ------------------------------------------------------------------
    # SCHEMATIC ROUTE MAP (simple diagram drawn from lat/lon)
    # ------------------------------------------------------------------
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Route Overview (schematic map)")
    y -= 0.5 * cm

    map_top = y
    map_height = 7 * cm
    map_bottom = map_top - map_height
    map_left = 2.0 * cm
    map_right = width - 2.0 * cm

    # Border for schematic map area
    c.setStrokeColor(HexColor("#cccccc"))
    c.rect(map_left, map_bottom, map_right - map_left, map_height, stroke=1, fill=0)

    # Collect coordinates
    lats = solution_df["latitude"].tolist()
    lons = solution_df["longitude"].tolist()
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    lat_range = max(max_lat - min_lat, 1e-6)
    lon_range = max(max_lon - min_lon, 1e-6)

    def to_map_xy(lat, lon):
        """Convert lat/lon to PDF schematic map coordinates."""
        x_norm = (lon - min_lon) / lon_range
        y_norm = (lat - min_lat) / lat_range
        x = map_left + x_norm * (map_right - map_left)
        y_coord = map_bottom + y_norm * map_height
        return x, y_coord

    # Color palette for vehicles
    vehicle_colors = [
        HexColor("#ff6347"),  # tomato
        HexColor("#1e90ff"),  # dodger blue
        HexColor("#228b22"),  # forest green
        HexColor("#ee82ee"),  # violet
        HexColor("#ffa500"),  # orange
        HexColor("#00ced1"),  # turquoise
        HexColor("#dc143c"),  # crimson
        HexColor("#9acd32"),  # yellow-green
    ]

    # Draw each vehicle's route
    c.setLineWidth(1)
    for vid in vehicles:
        color = vehicle_colors[int(vid) % len(vehicle_colors)]
        vdf = solution_df[solution_df["vehicle"] == vid].sort_values("stop_order")

        points = [
            to_map_xy(row["latitude"], row["longitude"])
            for _, row in vdf.iterrows()
        ]
        if len(points) < 2:
            continue

        # Route polyline
        c.setStrokeColor(color)
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            c.line(x1, y1, x2, y2)

        # Stops as small circles
        for (x, y_coord) in points:
            c.setFillColor(color)
            c.circle(x, y_coord, 1.5, stroke=0, fill=1)

    # Highlight depot (if present)
    depot_rows = solution_df[solution_df["name"] == "Depot"]
    if not depot_rows.empty:
        dlat = depot_rows.iloc[0]["latitude"]
        dlon = depot_rows.iloc[0]["longitude"]
        dx, dy = to_map_xy(dlat, dlon)
        c.setFillColor(HexColor("#ffeb3b"))  # yellow
        c.setStrokeColor(HexColor("#000000"))
        c.circle(dx, dy, 3, stroke=1, fill=1)

    y = map_bottom - 1.0 * cm

    # ------------------------------------------------------------------
    # VEHICLE ROUTE CARDS
    # ------------------------------------------------------------------
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(HexColor("#000000"))
    c.drawString(2 * cm, y, "Vehicle Routes")
    y -= 0.7 * cm

    c.setFont("Helvetica", 10)

    for vid in vehicles:
        vdf = solution_df[solution_df["vehicle"] == vid].sort_values("stop_order")
        stops = " → ".join(vdf["name"].tolist())
        load = int(vdf["demand"].sum())
        num_stops = (vdf["name"] != "Depot").sum()

        # New page if needed
        if y < 3 * cm:
            c.showPage()
            c.setFont("Helvetica-Bold", 12)
            c.drawString(2 * cm, height - 2 * cm, "Vehicle Routes (continued)")
            y = height - 3 * cm
            c.setFont("Helvetica", 10)

        c.setFont("Helvetica-Bold", 10)
        c.drawString(2 * cm, y, f"Vehicle {vid} (load {load})")
        y -= 0.4 * cm

        c.setFont("Helvetica", 9)
        c.drawString(2.5 * cm, y, f"Stops (excluding depot): {num_stops}")
        y -= 0.4 * cm

        route_text = f"Route: {stops}"
        max_chars = 115
        if len(route_text) > max_chars:
            route_text = route_text[: max_chars - 3] + "..."

        c.drawString(2.5 * cm, y, route_text)
        y -= 0.8 * cm

    # ------------------------------------------------------------------
    # FOOTER
    # ------------------------------------------------------------------
    c.setFont("Helvetica", 8)
    c.setFillColor(HexColor("#555555"))
    footer_text = "G-OPT | Routing & Optimization Demo"
    c.drawString(2 * cm, 1.3 * cm, footer_text)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


# ======================================================
# STREAMLIT APP
# ======================================================

st.set_page_config(page_title="G-OPT VRPTW Demo", layout="wide")

logo_path = Path("gopt_logo.png")

# ---- Professional left-aligned header ----
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
    """
Upload a CSV of your locations and constraints, choose your fleet settings,
and G-OPT will compute optimized routes with time windows and capacity constraints.
"""
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
    max_value=30,
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

# Google API key from Streamlit Secrets (for cloud deployment)
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
# Load CSV
# ======================================================

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("sample_locations.csv")

required_cols = {
    "name", "latitude", "longitude",
    "demand", "ready_time", "due_time", "service_time",
}

if not required_cols.issubset(df.columns):
    st.error(f"CSV must contain the following columns: {required_cols}")
    st.stop()

st.subheader("📍 Input Locations")
st.dataframe(df, use_container_width=True)

coords = list(zip(df["latitude"], df["longitude"]))
demands = df["demand"].tolist()
ready_times = df["ready_time"].tolist()
due_times = df["due_time"].tolist()
service_times = df["service_time"].tolist()

total_demand = sum(demands[1:])  # exclude depot
total_capacity = num_vehicles * vehicle_capacity

# Summary metrics
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
    with st.spinner("Running VRPTW solver..."):

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
        st.stop()

    st.success(f"Optimization complete! Total distance = **{total_km:.2f} km**")

    # --------------------------------------------------
    # Vehicle Routes (expanders)
    # --------------------------------------------------
    st.subheader("🧭 Vehicle Routes")

    solution_df = routes_to_dataframe(df, routes)
    vehicles = sorted(solution_df["vehicle"].unique().tolist())

    for vid in vehicles:
        vdf = solution_df[solution_df["vehicle"] == vid].sort_values("stop_order")
        stops = " → ".join(vdf["name"].tolist())
        load = int(vdf["demand"].sum())
        with st.expander(f"Vehicle {vid} (load {load}/{vehicle_capacity})"):
            st.write(stops)

    # --------------------------------------------------
    # Detailed Route Table + downloads
    # --------------------------------------------------
    st.subheader("📘 Detailed Route Table")
    st.dataframe(solution_df, use_container_width=True)

    csv_bytes = solution_df.to_csv(index=False).encode("utf-8")
    dl_cols = st.columns(2)
    with dl_cols[0]:
        st.download_button(
            "⬇ Download results as CSV",
            csv_bytes,
            "vrptw_solution.csv",
            "text/csv",
        )

    with dl_cols[1]:
        pdf_bytes = create_pdf_report(
            total_km=total_km,
            total_demand=total_demand,
            total_capacity=total_capacity,
            num_vehicles=num_vehicles,
            solution_df=solution_df,
            logo_path=logo_path if logo_path.exists() else None,
        )
        st.download_button(
            "⬇ Download PDF report",
            pdf_bytes,
            "gopt_vrptw_report.pdf",
            "application/pdf",
        )

    # ==================================================
    # Map Visualization
    # ==================================================
    st.subheader("🗺 Route Map (Google Roads)")

    sol_sorted = solution_df.sort_values(
        ["vehicle", "stop_order"]
    ).reset_index(drop=True)

    vehicles = sol_sorted["vehicle"].unique().tolist()

    st.markdown("### 🎨 Vehicle Legend & Visibility")

    route_colors = [
        [255, 99, 71],      # tomato
        [30, 144, 255],     # dodger blue
        [34, 139, 34],      # forest green
        [238, 130, 238],    # violet
        [255, 165, 0],      # orange
        [0, 206, 209],      # turquoise
        [220, 20, 60],      # crimson
        [154, 205, 50],     # yellow-green
    ]

    visible_vehicles = st.multiselect(
        "Show vehicles",
        options=vehicles,
        default=vehicles,
        format_func=lambda v: f"Vehicle {v}",
    )

    max_step = int(sol_sorted["stop_order"].max())
    if max_step < 1:
        max_step = 1

    anim_step = st.slider(
        "Animation step (max stop_order shown)",
        min_value=1,
        max_value=max_step,
        value=max_step,
    )

    layers = []

    # Depot highlight
    depot_df = df.iloc[0:1]
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=depot_df,
            get_position="[longitude, latitude]",
            get_radius=80,
            get_fill_color=[255, 230, 0],  # yellow
            pickable=True,
        )
    )

    if not depot_only:
        for vid in vehicles:
            if vid not in visible_vehicles:
                continue

            vdf_full = sol_sorted[sol_sorted["vehicle"] == vid]
            vdf = vdf_full[vdf_full["stop_order"] <= anim_step]

            if vdf.empty:
                continue

            color = route_colors[int(vid) % len(route_colors)]

            ordered_nodes = vdf[["latitude", "longitude"]].values.tolist()
            full_path = []

            for i in range(len(ordered_nodes) - 1):
                origin = (ordered_nodes[i][0], ordered_nodes[i][1])
                dest = (ordered_nodes[i + 1][0], ordered_nodes[i + 1][1])

                road_coords = None
                if use_google and api_key:
                    road_coords = get_route_polyline(origin, dest, api_key)

                if road_coords:
                    for lat, lon in road_coords:
                        full_path.append([lon, lat])
                else:
                    full_path.append([origin[1], origin[0]])
                    full_path.append([dest[1], dest[0]])

            # Path layer
            layers.append(
                pdk.Layer(
                    "PathLayer",
                    data=[{"path": full_path}],
                    get_path="path",
                    get_color=color,
                    width_scale=8,
                    width_min_pixels=2,
                )
            )

            # Node markers
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=vdf,
                    get_position="[longitude, latitude]",
                    get_radius=25,
                    get_fill_color=color,
                    pickable=True,
                )
            )

            # Labels
            if show_labels:
                layers.append(
                    pdk.Layer(
                        "TextLayer",
                        data=vdf,
                        get_position="[longitude, latitude]",
                        get_text="name",
                        get_size=16,
                        get_color=[60, 60, 60],
                        get_background=True,
                    )
                )

        # Legend text
        legend_lines = []
        for vid in vehicles:
            c = route_colors[int(vid) % len(route_colors)]
            color_box = (
                f"<span style='display:inline-block;width:12px;height:12px;"
                f"background:rgb({c[0]},{c[1]},{c[2]});margin-right:4px;"
                f"border-radius:2px;'></span>"
            )
            legend_lines.append(f"{color_box} Vehicle {vid}")
        legend_html = "<br>".join(legend_lines)
        st.markdown(f"**Legend:**<br>{legend_html}", unsafe_allow_html=True)
    else:
        st.info("Depot-only view is enabled (routes hidden).")

    midpoint = [
        float(df["longitude"].mean()),
        float(df["latitude"].mean()),
    ]

    view_state = pdk.ViewState(
        longitude=midpoint[0],
        latitude=midpoint[1],
        zoom=12,
    )

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=pdk.map_styles.LIGHT,
        tooltip={
            "text": "{name}\nDemand: {demand}\nWindow: {ready_time}-{due_time}"
        },
    )

    st.pydeck_chart(deck)
