# driver-schedule-builder-v3
Updated version of the schedule runs app with special priority drivers
# Driver Schedule Builder V3 – Special Driver Priority (Toronto 2025)

Advanced scheduling tool used daily by dispatch to create fair, efficient, and road-legal driver routes from hundreds of loose trips.

**V3 introduces priority scheduling for afternoon and limited-capacity drivers** — ensuring they get work first in their time windows and zones, preventing low/no KM days.

## Key Features

- **Special Driver Priority**  
  Afternoon/limited drivers are scheduled **first**, using only trips that match their:
  - Start time/window
  - Allowed zones
  - Custom KM limit (60km, 80km, 30–40km, etc.)
  - Custom shift length (e.g., 7-hour max from 14:00–19:00)

- **Custom Gap Rules for Special Drivers**  
  Longer randomized gaps when linking trips:
  - Distance 0: 10–20 min  
  - Distance 1: 20–30 min  
  - Distance 2: 30–40 min

- **Snow Mode (for regular drivers)**  
  Toggleable winter rules with longer gaps in northern zones (1,2,3,4,5,6,8,10,11,13,17,30,32,34)

- **Standard Rules (when Snow Mode off)**  
  - Distance 0 → 10 min  
  - Distance 1 → 15 min  
  - Distance 2 → 20 min

- Full audit trail with clear justifications for every trip link
- One-click export: CSV summary, full details, and combined Excel

## How It Works

1. Special drivers are processed **in order of start time**
2. Each gets the earliest available trip in their zones after their start time
3. Greedily adds compatible follow-up trips respecting their custom KM, hours, and extended gaps
4. Remaining trips → standard 120km / 12-hour schedules with normal or snow gaps

## Required Input Files

| File | Format | Required Columns / Notes |
|------|--------|--------------------------|
| Trips | CSV | `TTM Number`, `First Pickup Time` (HH:MM:SS), `Last Dropoff Time`, `First Pickup Zone`, `Last Dropoff Zone`, `KM` |
| Zone Graph | CSV or XLSX | `Primary Zone`, `Backup Zones` (comma-separated) |
| Special Drivers | CSV | `km`, `Driver`, `Time Start`, `Zones` (quoted comma-separated) <br> Examples supported: `60`, `30-40`, `14:00`, `14:00 to 19:00`, `12:00 PM` |

Example special drivers row:
```csv
60,A789,15:00,"132, 134, 130, 112, 110"
30-40,1012,14:00 to 19:00,"142, 140, 138, 136, 74, ..."
