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

    header_logo_w = 3.5 * cm
    header_logo_h = 3.5 * cm
    padding_top = height - 2.0 * cm

    if logo_path and logo_path.exists():
        try:
            c.drawImage(
                str(logo_path),
                2 * cm,
                padding_top - header_logo_h,
                width=header_logo_w,
                height=header_logo_h,
                preserveAspectRatio=True,
                mask="auto",
            )
        except Exception:
            pass

    title_x = 2 * cm + header_logo_w + 1 * cm
    title_y = padding_top - 0.5 * cm

    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(HexColor("#000000"))
    c.drawString(title_x, title_y, "G-OPT Route Optimization Report")

    c.setFont("Helvetica", 11)
    c.drawString(title_x, title_y - 0.8 * cm, "Professional VRPTW Summary Report")

    y = padding_top - header_logo_h - 1.5 * cm

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

    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Route Overview (schematic map)")
    y -= 0.5 * cm

    map_top = y
    map_height = 7 * cm
    map_bottom = map_top - map_height
    map_left = 2.0 * cm
    map_right = width - 2.0 * cm

    c.setStrokeColor(HexColor("#cccccc"))
    c.rect(map_left, map_bottom, map_right - map_left, map_height, stroke=1, fill=0)

    lats = solution_df["latitude"].tolist()
    lons = solution_df["longitude"].tolist()
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    lat_range = max(max_lat - min_lat, 1e-6)
    lon_range = max(max_lon - min_lon, 1e-6)

    def to_map_xy(lat, lon):
        x_norm = (lon - min_lon) / lon_range
        y_norm = (lat - min_lat) / lat_range
        x = map_left + x_norm * (map_right - map_left)
        y_coord = map_bottom + y_norm * map_height
        return x, y_coord

    vehicle_colors = [
        HexColor("#ff6347"), HexColor("#1e90ff"), HexColor("#228b22"), HexColor("#ee82ee"),
        HexColor("#ffa500"), HexColor("#00ced1"), HexColor("#dc143c"), HexColor("#9acd32"),
    ]

    c.setLineWidth(1)
    for vid in vehicles:
        color = vehicle_colors[int(vid) % len(vehicle_colors)]
        vdf = solution_df[solution_df["vehicle"] == vid].sort_values("stop_order")
        points = [to_map_xy(row["latitude"], row["longitude"]) for _, row in vdf.iterrows()]
        if len(points) < 2: continue
        c.setStrokeColor(color)
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            c.line(x1, y1, x2, y2)
        for (x, y_coord) in points:
            c.setFillColor(color)
            c.circle(x, y_coord, 1.5, stroke=0, fill=1)

    depot_rows = solution_df[solution_df["name"] == "Depot"]
    if not depot_rows.empty:
        dlat, dlon = depot_rows.iloc[0]["latitude"], depot_rows.iloc[0]["longitude"]
        dx, dy = to_map_xy(dlat, dlon)
        c.setFillColor(HexColor("#ffeb3b"))
        c.setStrokeColor(HexColor("#000000"))
        c.circle(dx, dy, 3, stroke=1, fill=1)

    y = map_bottom - 1.0 * cm
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
        if len(route_text) > 115: route_text = route_text[:112] + "..."
        c.drawString(2.5 * cm, y, route_text)
        y -= 0.8 * cm

    c.setFont("Helvetica", 8)
    c.setFillColor(HexColor("#555555"))
    c.drawString(2 * cm, 1.3 * cm, "G-OPT | Routing & Optimization Demo")
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


# ======================================================
# STREAMLIT APP
# ======================================================

st.set_page_config(page_title="G-OPT VRPTW Demo", layout="wide")
logo_path = Path("gopt_logo.png")

header_left, header_right = st.columns([1.2, 4])
with header_left:
    if logo_path.exists(): st.image(str(logo_path), width=110)
with header_right:
    st.markdown("## 🚚 G-OPT Route Optimization (VRPTW + Google Roads)")
    st.markdown("<span style='font-size:16px;'>Solve complex <b>VRPTW</b> using real Google Maps distances.</span>", unsafe_allow_html=True)

st.sidebar.header("⚙️ Solver Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
num_vehicles = st.sidebar.number_input("Number of vehicles", 1, 30, 3)
vehicle_capacity = st.sidebar.number_input("Vehicle capacity", 1, 1000, 7)
use_google = st.sidebar.checkbox("Use Google Maps (real road distances + paths)", value=True)

api_key = st.secrets.get("GOOGLE_API_KEY") if use_google else None
if use_google and not api_key:
    api_key = st.sidebar.text_input("Google API key", type="password")

st.sidebar.markdown("---")
st.sidebar.header("🗺 Map Options")
show_labels = st.sidebar.checkbox("Show customer labels", value=True)
depot_only = st.sidebar.checkbox("Show only depot (hide routes)", value=False)

df = pd.read_csv(uploaded_file if uploaded_file else "sample_locations.csv")
coords = list(zip(df["latitude"], df["longitude"]))
demands, ready_times, due_times, service_times = df["demand"].tolist(), df["ready_time"].tolist(), df["due_time"].tolist(), df["service_time"].tolist()

# Metrics
sum_cols = st.columns(4)
total_demand = sum(demands[1:])
total_capacity = num_vehicles * vehicle_capacity
sum_cols[0].metric("Total demand", total_demand)
sum_cols[1].metric("Total capacity", total_capacity)
sum_cols[2].metric("Utilization", f"{(total_demand/total_capacity*100):.1f} %" if total_capacity > 0 else "0%")
sum_cols[3].metric("Vehicles", num_vehicles)

if st.button("🚀 Optimize Routes"):
    with st.spinner("Running VRPTW solver..."):
        routes, total_km = solve_vrp(coords, demands, vehicle_capacity, ready_times, due_times, service_times, num_vehicles, use_google, api_key)
    
    if routes is None:
        st.error("❌ No feasible solution found.")
        st.stop()

    st.success(f"Optimization complete! Total distance = **{total_km:.2f} km**")
    solution_df = routes_to_dataframe(df, routes)
    
    # Downloads
    dl_cols = st.columns(2)
    dl_cols[0].download_button("⬇ Download CSV", solution_df.to_csv(index=False).encode("utf-8"), "solution.csv", "text/csv")
    pdf_bytes = create_pdf_report(total_km, total_demand, total_capacity, num_vehicles, solution_df, logo_path)
    dl_cols[1].download_button("⬇ Download PDF", pdf_bytes, "report.pdf", "application/pdf")

    # Map Visualization
    st.subheader("🗺 Route Map")
    sol_sorted = solution_df.sort_values(["vehicle", "stop_order"]).reset_index(drop=True)
    vehicles = sol_sorted["vehicle"].unique().tolist()
    
    visible_vehicles = st.multiselect("Show vehicles", options=vehicles, default=vehicles, format_func=lambda v: f"Vehicle {v}")
    anim_step = st.slider("Animation step", 1, int(sol_sorted["stop_order"].max()), int(sol_sorted["stop_order"].max()))

    route_colors = [[255, 99, 71], [30, 144, 255], [34, 139, 34], [238, 130, 238], [255, 165, 0], [0, 206, 209], [220, 20, 60], [154, 205, 50]]
    layers = []

    # Depot
    layers.append(pdk.Layer("ScatterplotLayer", data=df.iloc[0:1], get_position="[longitude, latitude]", get_radius=100, get_fill_color=[255, 230, 0], pickable=True))

    if not depot_only:
        # OPTIMIZATION: If n > 25, use straight lines (Geometric) to avoid API overhead and crashes
        is_large_dataset = len(coords) > 25

        for vid in visible_vehicles:
            vdf = sol_sorted[(sol_sorted["vehicle"] == vid) & (sol_sorted["stop_order"] <= anim_step)]
            if vdf.empty: continue
            color = route_colors[int(vid) % len(route_colors)]
            ordered_nodes = vdf[["latitude", "longitude"]].values.tolist()
            full_path = []

            for i in range(len(ordered_nodes) - 1):
                origin = (ordered_nodes[i][0], ordered_nodes[i][1])
                dest = (ordered_nodes[i + 1][0], ordered_nodes[i + 1][1])
                
                road_coords = None
                # Only request curvy road polylines if dataset is small
                if use_google and api_key and not is_large_dataset:
                    road_coords = get_route_polyline(origin, dest, api_key)

                if road_coords:
                    for lat, lon in road_coords: full_path.append([lon, lat])
                else:
                    # Fallback/Default for large datasets: Straight lines
                    full_path.append([origin[1], origin[0]])
                    full_path.append([dest[1], dest[0]])

            layers.append(pdk.Layer("PathLayer", data=[{"path": full_path}], get_path="path", get_color=color, width_scale=8, width_min_pixels=2))
            layers.append(pdk.Layer("ScatterplotLayer", data=vdf, get_position="[longitude, latitude]", get_radius=40, get_fill_color=color, pickable=True))
            if show_labels:
                layers.append(pdk.Layer("TextLayer", data=vdf, get_position="[longitude, latitude]", get_text="name", get_size=16, get_color=[0, 0, 0], get_background=True))

    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(longitude=df["longitude"].mean(), latitude=df["latitude"].mean(), zoom=6 if is_large_dataset else 11), map_style=pdk.map_styles.LIGHT))
