import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict, deque
from io import BytesIO
import random

# ==============================
# CONSTANTS
# ==============================

KM_LIMIT = 120.0
MAX_HOURS = 12.0
MAX_ZONE_DEPTH = 2

NORMAL_GAP = {0: 10, 1: 15, 2: 20}

SNOW_GAP_RANGES = {
    0: (10, 15),
    1: (15, 20),
    2: (20, 25)
}

SPECIAL_GAP_RANGES = {
    0: (10, 20),
    1: (20, 30),
    2: (30, 40)
}

SNOW_ZONES = {1, 2, 3, 4, 5, 6, 8, 10, 11, 13, 17, 30, 32, 34}

# ==============================
# HELPERS
# ==============================

def parse_time_str(s: str) -> datetime:
    return datetime.strptime(str(s).strip(), "%H:%M:%S")

def safe_read(file):
    file.seek(0)
    return BytesIO(file.read())

def load_trips(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(safe_read(uploaded_file))
    df["pickup_dt"] = df["First Pickup Time"].apply(parse_time_str)
    df["drop_dt"] = df["Last Dropoff Time"].apply(parse_time_str)
    df = df.sort_values("pickup_dt").reset_index(drop=True)
    return df

def load_zone_graph(uploaded_file) -> dict:
    file_bytes = safe_read(uploaded_file)
    if uploaded_file.name.lower().endswith(".csv"):
        zdf = pd.read_csv(file_bytes)
    else:
        zdf = pd.read_excel(file_bytes)

    neighbors = defaultdict(set)
    for _, row in zdf.iterrows():
        if pd.isna(row.get("Primary Zone")):
            continue
        p = int(row["Primary Zone"])
        neighbors[p].add(p)
        raw = "" if pd.isna(row.get("Backup Zones")) else str(row["Backup Zones"])
        for part in raw.split(","):
            part = part.strip()
            if part.isdigit():
                b = int(part)
                neighbors[p].add(b)
                neighbors[b].add(p)
    return dict(neighbors)

def load_special_drivers(uploaded_file) -> list:
    df = pd.read_csv(safe_read(uploaded_file))
    special = []
    for _, row in df.iterrows():
        if pd.isna(row.get('Driver')) or pd.isna(row.get('km')):
            continue

        name = str(row['Driver']).strip()
        km_str = str(row['km']).strip()
        km_limit = float(km_str.split('-')[1]) if '-' in km_str else float(km_str)

        time_str = str(row.get('Time Start', '')).strip()
        if not time_str:
            continue

        if 'to' in time_str.lower():
            parts = time_str.split(' to ')
            if len(parts) != 2:
                continue
            start_str, end_str = parts
            try:
                start_time = datetime.strptime(start_str.strip(), "%H:%M").time()
                end_time = datetime.strptime(end_str.strip(), "%H:%M").time()
                delta = datetime.combine(datetime.min, end_time) - datetime.combine(datetime.min, start_time)
                max_h = delta.total_seconds() / 3600
                if max_h < 0:
                    max_h += 24
            except ValueError:
                continue
        else:
            try:
                start_time = datetime.strptime(time_str, "%H:%M").time()
            except ValueError:
                try:
                    start_time = datetime.strptime(time_str, "%I:%M %p").time()
                except ValueError:
                    continue
            max_h = MAX_HOURS

        zones_str = str(row.get('Zones', '')).strip().strip('"')
        start_zones = set()
        if zones_str:
            try:
                start_zones = {int(z.strip()) for z in zones_str.split(',') if z.strip().isdigit()}
            except ValueError:
                start_zones = set()

        special.append({
            'name': name,
            'km_limit': km_limit,
            'start_time': start_time,
            'max_hours': max_h,
            'start_zones': start_zones,  # renamed for clarity
            'start_dt': datetime.combine(datetime.min, start_time)
        })

    special.sort(key=lambda d: d['start_dt'])
    return special

def zone_distance(neighbors: dict, start: int, target: int, max_depth: int = 2):
    start, target = int(start), int(target)
    if start == target:
        return 0
    visited = {start}
    q = deque([(start, 0)])
    while q:
        z, d = q.popleft()
        if d >= max_depth:
            continue
        for nb in neighbors.get(z, ()):
            if nb in visited:
                continue
            visited.add(nb)
            nd = d + 1
            if nb == target:
                return nd
            q.append((nb, nd))
    return None

# ==============================
# SCHEDULING
# ==============================

def build_schedules(trips: pd.DataFrame, neighbors: dict, snow_mode: bool, special_drivers: list) -> list:
    UNASSIGNED = set(trips.index)
    schedules = []

    # Special drivers first
    for driver in special_drivers:
        # First trip MUST start in one of their allowed start zones
        eligible_first = [
            i for i in UNASSIGNED
            if trips.loc[i, 'pickup_dt'].time() >= driver['start_time']
            and int(trips.loc[i, 'First Pickup Zone']) in driver['start_zones']
        ]
        if not eligible_first:
            continue

        eligible_first.sort(key=lambda i: trips.loc[i, 'pickup_dt'])
        idx = eligible_first[0]
        current_indices = [idx]
        UNASSIGNED.remove(idx)
        total_km = float(trips.loc[idx, 'KM'])
        first_pickup = trips.loc[idx, 'pickup_dt']

        # Now chain normally — any zone allowed as long as distance ≤2
        while True:
            if total_km >= driver['km_limit'] - 1e-6:
                break

            prev_drop_time = trips.loc[current_indices[-1], 'drop_dt']
            prev_drop_zone = int(trips.loc[current_indices[-1], 'Last Dropoff Zone'])
            candidates = []

            for i in UNASSIGNED:
                pick_zone = int(trips.loc[i, 'First Pickup Zone'])
                dist = zone_distance(neighbors, prev_drop_zone, pick_zone)
                if dist is None or dist > MAX_ZONE_DEPTH:
                    continue

                min_gap = random.randint(*SPECIAL_GAP_RANGES[dist])
                if trips.loc[i, 'pickup_dt'] < prev_drop_time + timedelta(minutes=min_gap):
                    continue

                duration_h = (trips.loc[i, 'drop_dt'] - first_pickup).total_seconds() / 3600
                if duration_h > driver['max_hours'] + 1e-6:
                    continue

                if total_km + float(trips.loc[i, 'KM']) > driver['km_limit'] + 1e-6:
                    continue

                candidates.append((i, trips.loc[i, 'pickup_dt']))

            if not candidates:
                break
            candidates.sort(key=lambda x: x[1])
            next_idx = candidates[0][0]
            current_indices.append(next_idx)
            UNASSIGNED.remove(next_idx)
            total_km += float(trips.loc[next_idx, 'KM'])

        if len(current_indices) >= 1:  # at least the first trip
            schedules.append({
                "id": f"SPECIAL-{driver['name']}",
                "trip_indices": current_indices,
                "driver": driver['name'],
                "is_special": True
            })

    # Regular drivers on remaining trips
    counter = 1
    while UNASSIGNED:
        idx = min(UNASSIGNED, key=lambda i: trips.loc[i, "pickup_dt"])
        current_indices = [idx]
        UNASSIGNED.remove(idx)
        total_km = float(trips.loc[idx, "KM"])
        first_pickup = trips.loc[idx, "pickup_dt"]

        while True:
            if total_km >= KM_LIMIT - 1e-6:
                break

            prev_drop_time = trips.loc[current_indices[-1], "drop_dt"]
            prev_drop_zone = int(trips.loc[current_indices[-1], "Last Dropoff Zone"])
            candidates = []

            for i in UNASSIGNED:
                pick_zone = int(trips.loc[i, "First Pickup Zone"])
                dist = zone_distance(neighbors, prev_drop_zone, pick_zone)
                if dist is None or dist > MAX_ZONE_DEPTH:
                    continue

                snow_affected = snow_mode and (prev_drop_zone in SNOW_ZONES or pick_zone in SNOW_ZONES)
                min_gap = random.randint(*SNOW_GAP_RANGES[dist]) if snow_affected else NORMAL_GAP[dist]

                if trips.loc[i, "pickup_dt"] < prev_drop_time + timedelta(minutes=min_gap):
                    continue

                duration_h = (trips.loc[i, "drop_dt"] - first_pickup).total_seconds() / 3600
                if duration_h > MAX_HOURS + 1e-6:
                    continue

                if total_km + float(trips.loc[i, "KM"]) > KM_LIMIT + 1e-6:
                    continue

                candidates.append((i, trips.loc[i, "pickup_dt"]))

            if not candidates:
                break
            candidates.sort(key=lambda x: x[1])
            next_idx = candidates[0][0]
            current_indices.append(next_idx)
            UNASSIGNED.remove(next_idx)
            total_km += float(trips.loc[next_idx, "KM"])

        schedules.append({
            "id": f"SCH-{counter:03d}",
            "trip_indices": current_indices,
            "driver": "General",
            "is_special": False
        })
        counter += 1

    return schedules

# ==============================
# OUTPUT TABLES (unchanged except minor clarity)
# ==============================

def build_summary(trips: pd.DataFrame, schedules: list) -> pd.DataFrame:
    rows = []
    for s in schedules:
        idxs = s["trip_indices"]
        km_total = sum(float(trips.loc[i, "KM"]) for i in idxs)
        pickups = [trips.loc[i, "pickup_dt"] for i in idxs]
        drops = [trips.loc[i, "drop_dt"] for i in idxs]
        rows.append({
            "Schedule_ID": s["id"],
            "Driver": s["driver"],
            "Trip_Count": len(idxs),
            "Total_KM": round(km_total, 3),
            "Start_Time": min(pickups).strftime("%H:%M"),
            "End_Time": max(drops).strftime("%H:%M"),
        })
    return pd.DataFrame(rows)

def build_details(trips: pd.DataFrame, schedules: list, neighbors: dict, snow_mode: bool) -> pd.DataFrame:
    rows = []
    for s in schedules:
        sid = s["id"]
        idxs = s["trip_indices"]
        cum_km = 0.0
        is_special = s["is_special"]

        for order, idx in enumerate(idxs, start=1):
            row = trips.loc[idx]
            cum_km += float(row["KM"])

            if order == 1:
                justification = "First trip (must start in driver's allowed zones)."
            else:
                prev_idx = idxs[order-2]
                prev_row = trips.loc[prev_idx]
                delta_min = int((row["pickup_dt"] - prev_row["drop_dt"]).total_seconds() / 60)
                dist = zone_distance(neighbors, int(prev_row["Last Dropoff Zone"]), int(row["First Pickup Zone"]))

                if is_special:
                    r = SPECIAL_GAP_RANGES[dist]
                    rule = f"{r[0]}-{r[1]} min (special driver)"
                else:
                    snow_link = snow_mode and (int(prev_row["Last Dropoff Zone"]) in SNOW_ZONES or int(row["First Pickup Zone"]) in SNOW_ZONES)
                    if snow_link:
                        r = SNOW_GAP_RANGES[dist]
                        rule = f"{r[0]}-{r[1]} min (snow)"
                    else:
                        rule = f"{NORMAL_GAP[dist]} min"

                justification = f"{delta_min} min gap · distance {dist} · {rule}"

            rows.append({
                "Schedule_ID": sid,
                "Driver": s["driver"],
                "Trip Order": order,
                "Run Number": row["TTM Number"],
                "Pickup Time": row["pickup_dt"].strftime("%H:%M"),
                "Pick Zone": int(row["First Pickup Zone"]),
                "Dropoff Zone": int(row["Last Dropoff Zone"]),
                "Dropoff Time": row["drop_dt"].strftime("%H:%M"),
                "Trip KM": round(float(row["KM"]), 3),
                "Schedule Total KM": round(cum_km, 3),
                "Linkage Justification": justification,
            })
    return pd.DataFrame(rows)

# ==============================
# STREAMLIT UI (unchanged except small note)
# ==============================

st.title("Driver Schedule Builder V3 – Special Driver Priority")

snow_mode = st.sidebar.checkbox("Snow Mode Active (for general drivers)", value=True)
if snow_mode:
    st.sidebar.success("Snow gaps active in northern zones")
else:
    st.sidebar.info("Normal gaps: 10 min (dist 0) • 15 min (dist 1) • 20 min (dist 2)")

if "neighbors" not in st.session_state:
    st.session_state.neighbors = None

if st.session_state.neighbors is None:
    zones_file = st.file_uploader("Upload zones file (CSV/XLSX) – upload once", type=["csv", "xlsx"])
    if zones_file:
        with st.spinner("Loading zone graph..."):
            st.session_state.neighbors = load_zone_graph(zones_file)
        st.success("Zone graph loaded and cached!")
else:
    st.info("✓ Zone graph already loaded")
    if st.button("Reload zone file"):
        st.session_state.neighbors = None
        st.rerun()

if "special_drivers" not in st.session_state:
    st.session_state.special_drivers = None

if st.session_state.special_drivers is None:
    st.markdown("### Special Drivers CSV")
    st.info("**Zones column = starting zones only** (can go anywhere).")

    template_csv = """km,Driver,Time Start,Zones
60,970,14:30,"132, 134, 130, 112, 110"
60,1292,15:00,"17, 19, 21, 7, 5, 15, 13, 11, 4"
60,A789,15:00,"132, 134, 130, 112, 110"
60,A1116,16:00,"35, 33, 65, 53, 51, 31, 15, 17, 19"
60,A0004,15:45,"35, 33, 55, 53, 51, 31, 15, 17, 19"
60,A1207,16:00,"47, 45, 43, 63, 81, 79, 123, 127"
80,1936,12:00 PM,"47, 45, 43, 63, 81, 79, 123, 127"
30-40,1012,14:00 to 19:00,"142, 140, 138, 136, 74, 72, 70, 16, 14, 412, 410, 58, 56, 40, 36, 34, 308, 306, 446, 444"
80,A224,12:00 PM,"131, 133, 113, 115, 111, 130, 110, 73, 71, 55, 53"
"""

    st.download_button(
        label="📥 Download Special Drivers Template CSV",
        data=template_csv,
        file_name="special_drivers_template.csv",
        mime="text/csv"
    )

    special_file = st.file_uploader("Upload your filled special drivers CSV", type="csv")
    if special_file:
        with st.spinner("Loading special drivers..."):
            st.session_state.special_drivers = load_special_drivers(special_file)
        count = len(st.session_state.special_drivers)
        st.success(f"Loaded {count} special driver{'s' if count != 1 else ''}!")
else:
    count = len(st.session_state.special_drivers)
    st.info(f"✓ {count} special driver{'s' if count != 1 else ''} loaded")
    if st.button("Reload special drivers file"):
        st.session_state.special_drivers = None
        st.rerun()

trips_file = st.file_uploader("Upload today's trips CSV", type="csv")

if st.button("Build Schedules", type="primary"):
    if not trips_file:
        st.warning("Please upload the trips CSV.")
    elif st.session_state.neighbors is None:
        st.warning("Please upload the zones file first.")
    elif st.session_state.special_drivers is None:
        st.warning("Please upload the special drivers CSV first.")
    else:
        with st.spinner("Building schedules – prioritizing special drivers..."):
            trips = load_trips(trips_file)
            schedules = build_schedules(
                trips,
                st.session_state.neighbors,
                snow_mode,
                st.session_state.special_drivers
            )
            summary_df = build_summary(trips, schedules)
            details_df = build_details(trips, schedules, st.session_state.neighbors, snow_mode)

        st.success(f"Done! Generated {len(schedules)} schedules")

        st.subheader("Schedule Summary")
        st.dataframe(summary_df, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button("Summary CSV", summary_df.to_csv(index=False).encode(),
                               "schedule_summary_v3.csv", "text/csv")
        with col2:
            st.download_button("Details CSV", details_df.to_csv(index=False).encode(),
                               "schedule_details_v3.csv", "text/csv")
        with col3:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                summary_df.to_excel(writer, sheet_name="Summary", index=False)
                details_df.to_excel(writer, sheet_name="Details", index=False)
            st.download_button("Both as Excel", output.getvalue(),
                               "driver_schedules_v3.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        with st.expander("View Full Details Table"):
            st.dataframe(details_df, use_container_width=True)

st.caption("Toronto Dispatch • Winter 2025 • Built with care for SCC")
