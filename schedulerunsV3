import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, time
from collections import defaultdict, deque
from io import BytesIO
import random

# ==============================
# CONSTANTS
# ==============================

KM_LIMIT = 120.0
MAX_HOURS = 12.0
MAX_ZONE_DEPTH = 2

# NORMAL (GOOD WEATHER) – FIXED gaps
NORMAL_GAP = {0: 10, 1: 15, 2: 20}

# SNOW MODE – randomized ranges
SNOW_GAP_RANGES = {
    0: (10, 15),
    1: (15, 20),
    2: (20, 25)
}

# SPECIAL DRIVERS – randomized gaps
SPECIAL_GAP_RANGES = {
    0: (10, 20),
    1: (20, 30),
    2: (30, 40)
}

# Snow zones
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
        if '-' in km_str:
            km_limit = float(km_str.split('-')[1])
        else:
            km_limit = float(km_str)
        time_str = str(row.get('Time Start', '')).strip()
        if not time_str:
            continue
        if 'to' in time_str:
            parts = time_str.split(' to ')
            if len(parts) != 2:
                continue
            start_str, end_str = parts
            try:
                start_time = datetime.strptime(start_str, "%H:%M").time()
                end_time = datetime.strptime(end_str, "%H:%M").time()
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
            max_h = 12.0
        zones_str = str(row.get('Zones', '')).strip().strip('"')
        zones = set()
        if zones_str:
            zones = set(int(z.strip()) for z in zones_str.split(',') if z.strip().isdigit())
        special.append({
            'name': name,
            'km_limit': km_limit,
            'start_time': start_time,
            'max_hours': max_h,
            'zones': zones,
            'start_dt': datetime.combine(datetime.min, start_time)
        })
    # Sort by start time
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
        # Eligible trips: start after driver start, pickup in zones
        eligible_indices = [
            i for i in UNASSIGNED
            if trips.loc[i, 'pickup_dt'].time() >= driver['start_time']
            and int(trips.loc[i, 'First Pickup Zone']) in driver['zones']
        ]
        if not eligible_indices:
            continue
        # Sort by pickup time
        eligible_indices.sort(key=lambda i: trips.loc[i, 'pickup_dt'])
        current_indices = []
        total_km = 0.0
        idx = eligible_indices[0]
        first_pickup = trips.loc[idx, 'pickup_dt']
        driver_km = driver['km_limit']
        driver_max_h = driver['max_hours']

        while True:
            current_indices.append(idx)
            UNASSIGNED.remove(idx)
            total_km += float(trips.loc[idx, 'KM'])
            eligible_indices = [e for e in eligible_indices if e != idx]  # remove
            if total_km >= driver_km - 1e-6:
                break
            prev_drop_time = trips.loc[idx, 'drop_dt']
            prev_drop_zone = int(trips.loc[idx, 'Last Dropoff Zone'])
            good_candidates = []
            for i in eligible_indices:
                pick_zone = int(trips.loc[i, 'First Pickup Zone'])
                dist = zone_distance(neighbors, prev_drop_zone, pick_zone, max_depth=MAX_ZONE_DEPTH)
                if dist is None or dist > MAX_ZONE_DEPTH:
                    continue
                # Special gap
                min_gap = random.randint(*SPECIAL_GAP_RANGES[dist])
                min_pickup_time = prev_drop_time + timedelta(minutes=min_gap)
                pick_time = trips.loc[i, 'pickup_dt']
                if pick_time < min_pickup_time:
                    continue
                # Hours check
                duration_hours = (trips.loc[i, 'drop_dt'] - first_pickup).total_seconds() / 3600.0
                if duration_hours > driver_max_h + 1e-6:
                    continue
                # KM check
                if total_km + float(trips.loc[i, 'KM']) > driver_km + 1e-6:
                    continue
                good_candidates.append((i, pick_time))
            if not good_candidates:
                break
            good_candidates.sort(key=lambda t: t[1])
            idx = good_candidates[0][0]

        if current_indices:
            schedules.append({
                "id": f"{driver['name']}-SCH",
                "trip_indices": current_indices,
                "driver": driver['name'],
                "is_special": True
            })

    # Regular schedules for remaining
    schedule_counter = 1
    while UNASSIGNED:
        earliest_idx = min(UNASSIGNED, key=lambda i: trips.loc[i, "pickup_dt"])
        current_indices = []
        total_km = 0.0
        current_idx = earliest_idx
        first_pickup = trips.loc[current_idx, "pickup_dt"]

        while True:
            current_indices.append(current_idx)
            UNASSIGNED.remove(current_idx)
            total_km += float(trips.loc[current_idx, "KM"])

            if total_km >= KM_LIMIT - 1e-6:
                break

            prev_drop_time = trips.loc[current_idx, "drop_dt"]
            prev_drop_zone = int(trips.loc[current_idx, "Last Dropoff Zone"])

            good_candidates = []

            for i in UNASSIGNED:
                pick_zone = int(trips.loc[i, "First Pickup Zone"])
                dist = zone_distance(neighbors, prev_drop_zone, pick_zone, max_depth=MAX_ZONE_DEPTH)
                if dist is None or dist > MAX_ZONE_DEPTH:
                    continue

                # Regular gap logic
                is_snow_affected = snow_mode and (prev_drop_zone in SNOW_ZONES or pick_zone in SNOW_ZONES)
                if is_snow_affected:
                    min_gap = random.randint(*SNOW_GAP_RANGES[dist])
                else:
                    min_gap = NORMAL_GAP[dist]

                min_pickup_time = prev_drop_time + timedelta(minutes=min_gap)
                pick_time = trips.loc[i, "pickup_dt"]
                if pick_time < min_pickup_time:
                    continue

                duration_hours = (trips.loc[i, "drop_dt"] - first_pickup).total_seconds() / 3600.0
                if duration_hours > MAX_HOURS + 1e-6:
                    continue

                if total_km + float(trips.loc[i, "KM"]) > KM_LIMIT + 1e-6:
                    continue

                good_candidates.append((i, pick_time))

            if not good_candidates:
                break

            good_candidates.sort(key=lambda t: t[1])
            current_idx = good_candidates[0][0]

        schedules.append({
            "id": f"SCH-{schedule_counter:03d}",
            "trip_indices": current_indices,
            "driver": "General",
            "is_special": False
        })
        schedule_counter += 1

    return schedules

def build_summary(trips: pd.DataFrame, schedules: list) -> pd.DataFrame:
    rows = []
    for s in schedules:
        idxs = s["trip_indices"]
        kmtotal = sum(float(trips.loc[i, "KM"]) for i in idxs)
        pickup_times = [trips.loc[i, "pickup_dt"] for i in idxs]
        drop_times = [trips.loc[i, "drop_dt"] for i in idxs]
        rows.append({
            "Schedule_ID": s["id"],
            "Driver": s["driver"],
            "Trip_Count": len(idxs),
            "Total_KM": round(kmtotal, 3),
            "Start_Time": min(pickup_times).strftime("%H:%M"),
            "End_Time": max(drop_times).strftime("%H:%M"),
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
            run_number = trips.loc[idx, "TTM Number"]
            pickup_dt = trips.loc[idx, "pickup_dt"]
            drop_dt = trips.loc[idx, "drop_dt"]
            pick_zone = int(trips.loc[idx, "First Pickup Zone"])
            drop_zone = int(trips.loc[idx, "Last Dropoff Zone"])
            km = float(trips.loc[idx, "KM"])
            cum_km += km

            if order == 1:
                justification = "First trip in schedule (earliest eligible for driver)."
            else:
                prev_idx = idxs[order - 2]
                prev_drop_time = trips.loc[prev_idx, "drop_dt"]
                prev_drop_zone = int(trips.loc[prev_idx, "Last Dropoff Zone"])
                delta_min = int((pickup_dt - prev_drop_time).total_seconds() / 60.0)
                dist = zone_distance(neighbors, prev_drop_zone, pick_zone, max_depth=MAX_ZONE_DEPTH)

                if is_special:
                    r = SPECIAL_GAP_RANGES[dist]
                    gap_rule = f"{r[0]}-{r[1]}-minute gap rule (special driver)"
                else:
                    is_snow_affected = snow_mode and (prev_drop_zone in SNOW_ZONES or pick_zone in SNOW_ZONES)
                    if is_snow_affected:
                        r = SNOW_GAP_RANGES[dist]
                        gap_rule = f"{r[0]}-{r[1]}-minute gap rule (snow)"
                    else:
                        gap_rule = f"{NORMAL_GAP[dist]}-minute gap rule"

                justification = (
                    f"Pickup {delta_min} mins after previous drop; "
                    f"pickup zone distance {dist} from dropoff zone {prev_drop_zone} "
                    f"({gap_rule})."
                )

            rows.append({
                "Schedule_ID": sid,
                "Driver": s["driver"],
                "Trip Order": order,
                "Run Number": run_number,
                "Pickup Time": pickup_dt.strftime("%H:%M"),
                "Pick Zone": pick_zone,
                "Dropoff Zone": drop_zone,
                "Dropoff Time": drop_dt.strftime("%H:%M"),
                "Trip KM": round(km, 3),
                "Schedule Total KM": round(cum_km, 3),
                "Linkage Justification": justification,
            })

    return pd.DataFrame(rows)

# ==============================
# STREAMLIT APP
# ==============================

st.title("Driver Schedule Builder V3 – Special Driver Priority")

# Snow mode toggle
snow_mode = st.sidebar.checkbox("Snow Mode Active (for general schedules)", value=True)
if snow_mode:
    st.sidebar.success("Snow gaps for general schedules in northern zones")
else:
    st.sidebar.info("Normal gaps for general: 10 (0), 15 (1), 20 (2) min")

# Zone graph – cached
if "neighbors" not in st.session_state:
    st.session_state.neighbors = None

if st.session_state.neighbors is None:
    zones_file = st.file_uploader("Upload zones file (CSV or XLSX) - upload once", type=["csv", "xlsx"])
    if zones_file:
        with st.spinner("Loading zone graph..."):
            st.session_state.neighbors = load_zone_graph(zones_file)
        st.success("Zone graph loaded!")
else:
    st.info("Zone graph loaded from previous upload.")
    if st.button("Reload zone file"):
        st.session_state.neighbors = None
        st.rerun()

# Special drivers – cached
if "special_drivers" not in st.session_state:
    st.session_state.special_drivers = None

if st.session_state.special_drivers is None:
    special_file = st.file_uploader("Upload special drivers CSV - upload once", type="csv")
    if special_file:
        with st.spinner("Loading special drivers..."):
            st.session_state.special_drivers = load_special_drivers(special_file)
        st.success(f"Loaded {len(st.session_state.special_drivers)} special drivers!")
else:
    st.info("Special drivers loaded from previous upload.")
    if st.button("Reload special drivers file"):
        st.session_state.special_drivers = None
        st.rerun()

# Trips file
trips_file = st.file_uploader("Upload today's trips CSV", type="csv")

if st.button("Build Schedules") and trips_file and st.session_state.neighbors and st.session_state.special_drivers is not None:
    with st.spinner("Building schedules with special priority..."):
        trips = load_trips(trips_file)
        neighbors = st.session_state.neighbors
        special_drivers = st.session_state.special_drivers
        schedules = build_schedules(trips, neighbors, snow_mode, special_drivers)
        summary_df = build_summary(trips, schedules)
        details_df = build_details(trips, schedules, neighbors, snow_mode)

    st.success(f"Generated {len(schedules)} schedules (including special drivers)!")

    # Summary table
    st.subheader("Schedule Summary")
    st.dataframe(summary_df)

    # Downloads
    csv_summary = summary_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Summary CSV",
        data=csv_summary,
        file_name='schedule_summary_v3.csv',
        mime='text/csv'
    )

    csv_details = details_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Details CSV",
        data=csv_details,
        file_name='full_schedule_details_v3.csv',
        mime='text/csv'
    )

    # Excel download
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        details_df.to_excel(writer, sheet_name='Details', index=False)
    output.seek(0)
    st.download_button(
        label="Download All as Excel",
        data=output,
        file_name='driver_schedules_v3.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

    # Optional: Display details
    with st.expander("View Details Table"):
        st.dataframe(details_df)

else:
    if not trips_file:
        st.warning("Please upload the trips CSV.")
    if st.session_state.neighbors is None:
        st.warning("Please upload the zones file first.")
    if st.session_state.special_drivers is None:
        st.warning("Please upload the special drivers CSV first.")
