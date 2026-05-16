"""Generate synthetic GrabWise data with realistic distributions + per-user history.

Run from project root:
    python -m backend.db.seed
"""
from __future__ import annotations
import math
import random
import re
from datetime import datetime, timedelta
from collections import defaultdict
from faker import Faker
from sqlalchemy import select


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points, in kilometers."""
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

from backend.db.database import init_db, get_session, engine, Base
from backend.db import models as M
from backend.api.auth import hash_password
from backend.tools._risk_math import (
    compute_customer_anomaly_score,
    compute_order_risk_score,
)

# Default fallback Faker (used for misc data — names use locale-specific instances below).
fake = Faker()
random.seed(42)
Faker.seed(42)

# --- Locale-aware Fakers per city -------------------------------------------
# Use a region-appropriate name pool per city so customers/drivers/merchants
# have culturally plausible names. Not every "ideal" locale exists in Faker
# (e.g. en_SG / ms_MY are not shipped) — we fall back gracefully.
#
# Faker built-in locales used here: id_ID, th_TH, en_PH, zh_CN, en_GB, en_US.
_CITY_LOCALE: dict[str, str] = {
    "Singapore":     "en_GB",   # no en_SG in Faker; en_GB is closest Anglo register
    "Jakarta":       "id_ID",
    "Bangkok":       "th_TH",
    "Manila":        "en_PH",
    "Kuala Lumpur":  "en_GB",   # no ms_MY in Faker; en_GB used (Malay names mixed in via fallback)
}
_CITY_PHONE_PREFIX: dict[str, str] = {
    "Singapore":     "+65",
    "Jakarta":       "+62",
    "Bangkok":       "+66",
    "Manila":        "+63",
    "Kuala Lumpur":  "+60",
}
_locale_fakers: dict[str, Faker] = {}
def _faker_for(city_name: str) -> Faker:
    """Return a memoized Faker instance for the given city's locale.

    Falls back to en_US if the chosen locale isn't supported by the installed
    Faker version — keeps the seed resilient across environments.
    """
    locale = _CITY_LOCALE.get(city_name, "en_US")
    f = _locale_fakers.get(locale)
    if f is None:
        try:
            f = Faker(locale)
        except (AttributeError, ValueError, Exception):
            print(f"  ⚠ Faker locale {locale!r} not available — falling back to en_US for {city_name!r}")
            f = Faker("en_US")
        Faker.seed(42)
        _locale_fakers[locale] = f
    return f


def _slugify_name(name: str) -> str:
    """Turn 'Aiden Tan' → 'aiden.tan' (deterministic, ASCII-safe-ish)."""
    cleaned = re.sub(r"[^A-Za-z0-9 ]+", "", name).strip().lower()
    parts = [p for p in cleaned.split() if p]
    return ".".join(parts[:3]) if parts else "user"


def _phone_for(city_name: str) -> str:
    prefix = _CITY_PHONE_PREFIX.get(city_name, "+65")
    return f"{prefix}{random.randint(80000000, 99999999)}"


def _avatar_url(seed: str) -> str:
    """Stable initials-style avatar via DiceBear's free API. Frontend can ignore + show initials."""
    safe = re.sub(r"[^A-Za-z0-9]+", "", seed) or "user"
    return f"https://api.dicebear.com/7.x/initials/svg?seed={safe}&backgroundColor=00B14F&textColor=ffffff"


def _email_for(name: str, role: str, idx: int, domain: str = "grabwise.demo") -> str:
    """e.g. 'Aiden Tan' + role='driver' + idx=42 → 'aiden.tan+driver42@grabwise.demo'."""
    local = _slugify_name(name)
    return f"{local}+{role}{idx}@{domain}"

# --- Static reference data: 5 cities, 4 zones each ----------------------------
CITIES = [
    ("Singapore",  "Singapore",   ["Orchard", "Bugis", "Bukit Timah", "Tampines"]),
    ("Jakarta",    "Indonesia",   ["Kemang", "Senayan", "Menteng", "Kelapa Gading"]),
    ("Bangkok",    "Thailand",    ["Sukhumvit", "Silom", "Ari", "Thonglor"]),
    ("Manila",     "Philippines", ["Makati", "BGC", "Ortigas", "Quezon City"]),
    ("Kuala Lumpur", "Malaysia",  ["Bangsar", "Mont Kiara", "KLCC", "Bukit Bintang"]),
]
CUISINES = ["Local", "Chinese", "Thai", "Indian", "Japanese", "Western", "Vegetarian", "Korean"]
DIETS = [["vegetarian"], ["vegan"], ["halal"], ["gluten-free"], [], [], []]

CUISINE_MENU: dict[str, list[tuple[str, float, list[str]]]] = {
    "Local": [
        ("Hainanese Chicken Rice", 7.50, ["halal"]),
        ("Laksa", 8.50, []),
        ("Char Kway Teow", 7.00, []),
        ("Roti Prata", 5.00, ["vegetarian", "halal"]),
        ("Nasi Lemak", 6.50, ["halal"]),
        ("Bak Kut Teh", 11.00, []),
        ("Satay (10 sticks)", 12.00, ["halal"]),
        ("Mee Goreng", 7.00, ["vegetarian"]),
        ("Popiah", 6.00, ["vegetarian"]),
        ("Hokkien Mee", 8.00, []),
    ],
    "Chinese": [
        ("Sweet & Sour Pork", 14.00, []),
        ("Mapo Tofu", 12.00, ["vegetarian"]),
        ("Kung Pao Chicken", 14.50, []),
        ("Yangzhou Fried Rice", 10.00, []),
        ("Wonton Noodles", 9.50, []),
        ("Dim Sum Platter", 18.00, []),
        ("Sichuan Hot Pot Bowl", 22.00, []),
        ("Spring Rolls (4 pcs)", 7.00, ["vegetarian"]),
        ("Char Siu Rice", 11.00, []),
        ("Steamed Pak Choi", 8.00, ["vegetarian", "vegan", "gluten-free"]),
    ],
    "Thai": [
        ("Pad Thai", 11.00, []),
        ("Green Curry Chicken", 13.50, []),
        ("Tom Yum Soup", 9.50, []),
        ("Massaman Curry", 14.00, []),
        ("Mango Sticky Rice", 8.00, ["vegetarian", "gluten-free"]),
        ("Pad See Ew", 11.00, []),
        ("Som Tam (Papaya Salad)", 9.00, ["vegan", "gluten-free"]),
        ("Basil Chicken Rice", 10.50, []),
        ("Tofu Pad Thai", 11.00, ["vegetarian", "vegan"]),
        ("Pineapple Fried Rice", 12.00, ["vegetarian"]),
    ],
    "Indian": [
        ("Butter Chicken", 14.50, ["halal"]),
        ("Chicken Biryani", 13.00, ["halal"]),
        ("Palak Paneer", 12.00, ["vegetarian", "halal"]),
        ("Tandoori Chicken", 15.00, ["halal", "gluten-free"]),
        ("Garlic Naan", 4.00, ["vegetarian", "halal"]),
        ("Chana Masala", 10.00, ["vegetarian", "vegan", "halal"]),
        ("Samosa (3 pcs)", 6.00, ["vegetarian", "halal"]),
        ("Dal Tadka", 9.00, ["vegetarian", "vegan", "halal", "gluten-free"]),
        ("Paneer Tikka", 13.00, ["vegetarian", "halal", "gluten-free"]),
        ("Vegetable Biryani", 11.00, ["vegetarian", "halal"]),
    ],
    "Japanese": [
        ("Salmon Sashimi", 18.00, ["gluten-free"]),
        ("Chicken Teriyaki Bento", 14.00, []),
        ("Tonkatsu Set", 15.00, []),
        ("Tonkotsu Ramen", 13.00, []),
        ("California Roll", 12.00, []),
        ("Tempura Udon", 14.00, []),
        ("Yaki Udon", 12.00, []),
        ("Gyoza (5 pcs)", 8.00, []),
        ("Miso Soup", 4.00, ["vegetarian", "vegan", "gluten-free"]),
        ("Vegetable Tempura", 11.00, ["vegetarian"]),
    ],
    "Western": [
        ("Cheeseburger", 13.00, []),
        ("Caesar Salad", 11.00, ["vegetarian"]),
        ("Spaghetti Bolognese", 14.00, []),
        ("Grilled Chicken", 16.00, ["gluten-free"]),
        ("BLT Sandwich", 10.00, []),
        ("Ribeye Steak", 28.00, ["gluten-free"]),
        ("Margherita Pizza", 16.00, ["vegetarian"]),
        ("Truffle Fries", 9.00, ["vegetarian", "vegan"]),
        ("Mushroom Risotto", 15.00, ["vegetarian"]),
        ("Quinoa Power Bowl", 13.00, ["vegetarian", "vegan", "gluten-free"]),
    ],
    "Vegetarian": [
        ("Buddha Bowl", 12.00, ["vegetarian", "vegan", "gluten-free"]),
        ("Falafel Wrap", 10.00, ["vegetarian", "vegan"]),
        ("Veggie Burger", 12.00, ["vegetarian"]),
        ("Mushroom Risotto", 14.00, ["vegetarian"]),
        ("Quinoa Salad", 11.00, ["vegetarian", "vegan", "gluten-free"]),
        ("Hummus & Pita", 9.00, ["vegetarian", "vegan"]),
        ("Stir-Fried Vegetables", 10.00, ["vegetarian", "vegan", "gluten-free"]),
        ("Tofu Green Curry", 12.50, ["vegetarian", "vegan"]),
        ("Avocado Toast", 9.50, ["vegetarian"]),
        ("Tempeh Bibimbap", 13.00, ["vegetarian", "vegan"]),
    ],
    "Korean": [
        ("Bibimbap", 12.50, []),
        ("Bulgogi Beef Bowl", 14.00, []),
        ("Korean Fried Chicken", 16.00, []),
        ("Kimchi Stew (Jjigae)", 13.00, []),
        ("Japchae Glass Noodles", 11.00, []),
        ("Tteokbokki (Spicy Rice Cakes)", 10.00, ["vegetarian"]),
        ("Korean BBQ Set", 26.00, []),
        ("Kimbap Roll", 8.00, []),
        ("Soft Tofu Stew", 12.00, ["vegetarian"]),
        ("Vegetable Bibimbap", 11.00, ["vegetarian", "vegan"]),
    ],
}

ZONE_WEIGHTS = [0.45, 0.30, 0.15, 0.10]
HISTORY_DAYS = 120

# ----------------------------------------------------------------------------
# SPATIAL GRID — real-ish city bounding boxes for the 10×10 cell decomposition.
# (min_lat, min_lon, max_lat, max_lon)
# ----------------------------------------------------------------------------
CITY_BBOX: dict[str, tuple[float, float, float, float]] = {
    "Singapore":     (1.205, 103.620, 1.470, 103.990),
    "Jakarta":       (-6.400, 106.700, -6.100, 107.000),
    "Bangkok":       (13.640, 100.420, 13.940, 100.780),
    "Manila":        (14.450, 120.910, 14.760, 121.100),
    "Kuala Lumpur":  (3.050,  101.620, 3.250, 101.800),
}

# 10×10 grid per city
GRID_SIDE = 10

# Cell-type probability weights (per city, balanced commercial:residential:transit:industrial)
CELL_TYPE_DISTRIBUTION = [
    ("commercial",   0.32),
    ("residential",  0.45),
    ("transit",      0.13),
    ("industrial",   0.10),
]

# When orders are placed into cells, bias toward commercial > transit > residential > industrial
CELL_TYPE_ORDER_WEIGHT = {
    "commercial":   3.0,
    "transit":      2.0,
    "residential":  1.0,
    "industrial":   0.5,
}

# Per-zone cell quadrant mapping inside each city's 10×10 grid.
# Each city's 4 named zones are assigned to a 5×5 quadrant of the grid so that
# legacy zone-level queries still work AND each zone owns ~25 cells.
def _cell_zone(city_zones: list[str], grid_x: int, grid_y: int) -> str:
    """Map a (grid_x, grid_y) cell to one of the city's 4 zone names."""
    if not city_zones:
        return "Unknown"
    # Split 10×10 into 4 quadrants of 5×5 each:
    #   zone[0] = top-left   (gx<5, gy<5)
    #   zone[1] = top-right  (gx>=5, gy<5)
    #   zone[2] = bot-left   (gx<5, gy>=5)
    #   zone[3] = bot-right  (gx>=5, gy>=5)
    idx = (1 if grid_x >= GRID_SIDE // 2 else 0) + (2 if grid_y >= GRID_SIDE // 2 else 0)
    return city_zones[min(idx, len(city_zones) - 1)]

CUSTOMER_PERSONAS = [
    "late-night-orderer", "lunch-regular", "weekend-foodie",
    "vegetarian-explorer", "new-user", "high-spender",
]
DRIVER_PERSONAS = [
    "weekend-warrior", "lunch-rush", "low-cancel-veteran",
    "struggling-newbie", "late-night-only", "steady-allrounder",
]
MERCHANT_PERSONAS = [
    "lunch-dominant", "dinner-dominant", "weekend-spike",
    "declining-weekends", "rising-star", "balanced",
]

# Default password for every seeded persona. Hackathon-only.
DEFAULT_PASSWORD = "password"


def _weighted_zone(zones: list[str]) -> str:
    return random.choices(zones, weights=ZONE_WEIGHTS, k=1)[0]


def _hour_for_customer(persona: str) -> int:
    if persona == "late-night-orderer":
        return random.choices(
            [22, 23, 0, 1, 2, 3, 11, 12, 19, 20],
            weights=[10, 12, 9, 6, 4, 3, 1, 1, 2, 2],
        )[0]
    if persona == "lunch-regular":
        return random.choices(
            list(range(11, 14)) + list(range(18, 21)),
            weights=[8, 10, 8, 1, 1, 1],
        )[0]
    if persona == "weekend-foodie":
        return random.choices(list(range(11, 22)), weights=[2, 4, 4, 2, 2, 4, 6, 6, 5, 3, 2])[0]
    return _peak_hour_default()


def _peak_hour_default() -> int:
    buckets = (
        list(range(0, 7))
        + list(range(7, 11))
        + list(range(11, 14)) * 4
        + list(range(14, 18))
        + list(range(18, 22)) * 5
        + list(range(22, 24))
    )
    return random.choice(buckets)


def _hour_for_driver(persona: str) -> int:
    if persona == "lunch-rush":
        return random.choices(list(range(11, 14)) + list(range(18, 21)), weights=[8, 10, 8, 2, 2, 2])[0]
    if persona == "late-night-only":
        return random.choices([22, 23, 0, 1, 2, 3], weights=[6, 8, 6, 4, 3, 2])[0]
    if persona == "weekend-warrior":
        return random.choices(list(range(18, 24)) + list(range(11, 14)), weights=[6, 8, 6, 4, 3, 2, 3, 4, 3])[0]
    return _peak_hour_default()


def _hour_for_merchant(persona: str) -> int:
    if persona == "lunch-dominant":
        return random.choices(list(range(11, 14)) + list(range(18, 21)), weights=[10, 12, 8, 2, 3, 2])[0]
    if persona == "dinner-dominant":
        return random.choices(list(range(11, 14)) + list(range(18, 22)), weights=[2, 3, 2, 8, 12, 10, 6])[0]
    return _peak_hour_default()


def _customer_timestamp(customer_persona: str) -> datetime:
    if customer_persona == "new-user":
        days_ago = random.randint(0, 13)
    else:
        days_ago = random.randint(0, HISTORY_DAYS - 1)
    base = datetime.utcnow() - timedelta(days=days_ago)
    if customer_persona == "weekend-foodie" and base.weekday() < 4 and random.random() < 0.7:
        shift = (4 + random.randint(0, 2) - base.weekday()) % 7
        base = base + timedelta(days=shift)
    hour = _hour_for_customer(customer_persona)
    return base.replace(hour=hour, minute=random.randint(0, 59), second=0, microsecond=0)


def _gauss_rating(mean: float = 4.6, sd: float = 0.25, lo: float = 3.4, hi: float = 5.0) -> float:
    return round(max(lo, min(hi, random.gauss(mean, sd))), 2)


def _customer_dietary(persona: str) -> list[str]:
    if persona == "vegetarian-explorer":
        return random.choice([["vegetarian"], ["vegan"], ["vegetarian"]])
    return random.choice(DIETS)


def reset_and_seed() -> None:
    print("⏳ Dropping + creating tables...")
    Base.metadata.drop_all(bind=engine)
    init_db()

    with get_session() as s:
        # ------- Cities -------
        cities: list[M.City] = []
        for name, country, zones in CITIES:
            c = M.City(name=name, country=country, zones=zones)
            s.add(c); cities.append(c)
        s.flush()
        print(f"  ✓ {len(cities)} cities")

        # ------- Grid cells — 10×10 per city = 500 cells total ------------------
        # Each cell gets a center lat/lon within the city's bounding box,
        # is mapped to one of the city's 4 legacy zones (via quadrant), and
        # is tagged with a cell_type that biases its demand realism.
        cells_by_city: dict[int, list[M.GridCell]] = defaultdict(list)
        cells_by_city_zone: dict[tuple[int, str], list[M.GridCell]] = defaultdict(list)
        total_cells = 0
        for city in cities:
            bbox = CITY_BBOX.get(city.name)
            if not bbox:
                continue
            min_lat, min_lon, max_lat, max_lon = bbox
            lat_step = (max_lat - min_lat) / GRID_SIDE
            lon_step = (max_lon - min_lon) / GRID_SIDE
            zones = city.zones or []
            type_pool = []
            for name, weight in CELL_TYPE_DISTRIBUTION:
                type_pool.extend([name] * int(weight * 100))
            for gy in range(GRID_SIDE):
                for gx in range(GRID_SIDE):
                    # Cell center: midpoint of the cell's lat/lon range
                    center_lat = min_lat + (gy + 0.5) * lat_step
                    center_lon = min_lon + (gx + 0.5) * lon_step
                    zone_name = _cell_zone(zones, gx, gy)
                    # Center cells lean commercial; edges lean residential
                    near_center = (3 <= gx <= 6) and (3 <= gy <= 6)
                    if near_center and random.random() < 0.55:
                        cell_type = "commercial"
                    else:
                        cell_type = random.choice(type_pool)
                    cell = M.GridCell(
                        city_id=city.id,
                        grid_x=gx,
                        grid_y=gy,
                        center_lat=round(center_lat, 6),
                        center_lon=round(center_lon, 6),
                        zone_name=zone_name,
                        cell_type=cell_type,
                    )
                    s.add(cell)
                    cells_by_city[city.id].append(cell)
                    cells_by_city_zone[(city.id, zone_name)].append(cell)
                    total_cells += 1
        s.flush()
        print(f"  ✓ {total_cells} grid cells ({GRID_SIDE}×{GRID_SIDE} per city)")

        # ------- Drivers (200) — locale-aware names + city-coded phones -------
        drivers: list[M.Driver] = []
        city_weights = [0.30, 0.25, 0.18, 0.15, 0.12]
        for i in range(200):
            persona = random.choice(DRIVER_PERSONAS)
            city = random.choices(cities, weights=city_weights, k=1)[0]
            if persona == "low-cancel-veteran":
                rating = _gauss_rating(mean=4.85, sd=0.12)
                cancel = round(max(0.0, random.gauss(0.02, 0.01)), 3)
                joined = fake.date_time_between(start_date="-3y", end_date="-1y")
            elif persona == "struggling-newbie":
                rating = _gauss_rating(mean=4.0, sd=0.25)
                cancel = round(max(0.05, random.gauss(0.18, 0.05)), 3)
                joined = fake.date_time_between(start_date="-90d", end_date="-1d")
            else:
                rating = _gauss_rating()
                cancel = round(max(0.0, random.gauss(0.06, 0.04)), 3)
                joined = fake.date_time_between(start_date="-2y", end_date="-1d")
            local_fake = _faker_for(city.name)
            # home_zone — biased toward the busier zones in the city
            home_zone = _weighted_zone(city.zones)
            d = M.Driver(
                name=local_fake.name(),
                phone=_phone_for(city.name),
                city_id=city.id,
                vehicle_type=random.choices(["bike", "car"], weights=[0.7, 0.3])[0],
                rating=rating,
                cancel_rate=cancel,
                joined_date=joined,
                is_active=True,
                behavior_persona=persona,
                home_zone=home_zone,
            )
            s.add(d); drivers.append(d)
        s.flush()
        print(f"  ✓ {len(drivers)} drivers")

        # ------- Driver planner data (preferences + schedules + active sessions) -------
        # Persona-aware so the daily planner has realistic inputs to optimize against.
        #
        # Schedule templates: (day_of_week_set, start_hour, end_hour)
        # day_of_week 0=Mon, 6=Sun
        SCHEDULE_TEMPLATES_BY_PERSONA: dict[str, list[tuple[list[int], int, int]]] = {
            "low-cancel-veteran":  [([0, 1, 2, 3, 4], 17, 22), ([5, 6], 11, 23)],
            "weekend-warrior":     [([4], 17, 23), ([5, 6], 10, 23)],
            "lunch-rush":          [([0, 1, 2, 3, 4], 11, 14), ([0, 1, 2, 3, 4], 18, 21)],
            "late-night-only":     [([0, 1, 2, 3, 4, 5, 6], 22, 23), ([0, 1, 2, 3, 4, 5, 6], 0, 4)],
            "steady-allrounder":   [([0, 1, 2, 3, 4], 8, 17)],
            "struggling-newbie":   [([random.choice([0, 1, 2]), random.choice([3, 4]), random.choice([5, 6])], 10, 16)],
        }

        # Blackout hours templates by persona (HARD constraint in planner)
        BLACKOUT_BY_PERSONA: dict[str, list[int]] = {
            "late-night-only":   list(range(5, 21)),         # avoid daytime
            "lunch-rush":        [14, 15, 16, 17, 21, 22, 23],  # only lunch + dinner
            # Other personas: empty (no hard blackouts)
        }

        # Weekly earning target by persona (SGD-equivalent)
        WEEKLY_TARGET_BY_PERSONA: dict[str, float] = {
            "low-cancel-veteran":  700.0,
            "weekend-warrior":     520.0,
            "lunch-rush":          480.0,
            "late-night-only":     560.0,
            "steady-allrounder":   600.0,
            "struggling-newbie":   280.0,
        }

        prefs_count = 0
        sched_count = 0
        for d in drivers:
            persona = d.behavior_persona or "steady-allrounder"
            city = next(c for c in cities if c.id == d.city_id)

            # Preferred zones: home_zone + 1 random other zone in the city, biased
            # toward popular zones
            other_zones = [z for z in city.zones if z != d.home_zone]
            extra = random.choices(other_zones, k=1) if other_zones and random.random() < 0.65 else []
            preferred_zones = [d.home_zone] + extra

            prefs = M.DriverPreferences(
                driver_id=d.id,
                preferred_zones=preferred_zones,
                blackout_hours=BLACKOUT_BY_PERSONA.get(persona, []),
                weekly_target_sgd=WEEKLY_TARGET_BY_PERSONA.get(persona, 500.0),
                notify_plan_changes=True,
            )
            s.add(prefs); prefs_count += 1

            # Schedule rows — explode template into one row per (day_of_week)
            for days, start_h, end_h in SCHEDULE_TEMPLATES_BY_PERSONA.get(persona, [([0,1,2,3,4], 9, 18)]):
                for dow in days:
                    s.add(M.DriverSchedule(
                        driver_id=d.id,
                        day_of_week=dow,
                        start_hour=start_h,
                        end_hour=end_h,
                    ))
                    sched_count += 1

            # ~30% of drivers are currently ONLINE — open session for demo variety
            if random.random() < 0.30:
                start_minutes_ago = random.randint(15, 180)
                s.add(M.DriverActiveSession(
                    driver_id=d.id,
                    started_at=datetime.utcnow() - timedelta(minutes=start_minutes_ago),
                    ended_at=None,
                    end_reason=None,
                    resume_at=None,
                ))

        print(f"  ✓ {prefs_count} driver-preference rows")
        print(f"  ✓ {sched_count} driver-schedule rows ({sched_count // len(drivers)} avg days/driver)")

        # ------- Merchants (50) — locale-aware owner names -------
        merchants: list[M.Merchant] = []
        for _ in range(50):
            persona = random.choice(MERCHANT_PERSONAS)
            city = random.choice(cities)
            local_fake = _faker_for(city.name)
            owner_first = local_fake.first_name()
            shop_word = random.choice(['Kitchen', 'Cafe', 'Eats', 'Bites', 'Grill', 'House'])
            m = M.Merchant(
                name=f"{owner_first}'s {shop_word}",
                city_id=city.id,
                cuisine=random.choice(CUISINES),
                rating=_gauss_rating(mean=4.4, sd=0.3),
                avg_prep_min=random.randint(8, 25),
                zone=_weighted_zone(city.zones),
                behavior_persona=persona,
            )
            s.add(m); merchants.append(m)
        s.flush()
        print(f"  ✓ {len(merchants)} merchants")

        # ------- Menu items -------
        items_created = 0
        for merch in merchants:
            pool = CUISINE_MENU.get(merch.cuisine, CUISINE_MENU["Local"])
            chosen = random.sample(pool, k=min(8, len(pool)))
            for name, base_price, tags in chosen:
                actual_price = round(base_price * random.uniform(0.85, 1.25), 2)
                s.add(M.MenuItem(
                    merchant_id=merch.id,
                    name=name,
                    description=f"{merch.name}'s {name.lower()}",
                    price=actual_price,
                    tags=list(tags),
                    popularity=random.randint(0, 200),
                ))
                items_created += 1
        print(f"  ✓ {items_created} menu items")

        # ------- Customers (50) — locale-aware names -------
        customers: list[M.Customer] = []
        for i in range(50):
            persona = random.choice(CUSTOMER_PERSONAS)
            city = random.choice(cities)
            if persona == "new-user":
                signup = fake.date_time_between(start_date="-13d", end_date="-1d")
            else:
                signup = fake.date_time_between(start_date="-1y", end_date="-1d")
            local_fake = _faker_for(city.name)
            c = M.Customer(
                name=local_fake.name(),
                city_id=city.id,
                dietary_prefs=_customer_dietary(persona),
                signup_date=signup,
                behavior_persona=persona,
            )
            s.add(c); customers.append(c)
        s.flush()
        print(f"  ✓ {len(customers)} customers")

        # ------- Orders (~60000) — driven by customer persona for realism -------
        # Tuned for cell-level granularity: at 10×10 cells × 24h × 7dows = 16.8K
        # buckets per city, 60K orders gives ~10-15 orders in the busiest buckets,
        # enough signal for the DP planner's cell ranking to be stable.
        TARGET_ORDERS = 60000
        # Per-persona weights for how often a customer orders.
        cust_weight = {
            "high-spender":          3.0,
            "lunch-regular":         2.4,
            "late-night-orderer":    1.8,
            "weekend-foodie":        1.5,
            "vegetarian-explorer":   1.3,
            "new-user":              0.4,
        }
        c_weights = [cust_weight.get(c.behavior_persona, 1.0) for c in customers]

        # Per-driver weights (struggling newbies get fewer orders).
        drv_weight = {
            "low-cancel-veteran":  2.5,
            "weekend-warrior":     1.6,
            "lunch-rush":          1.6,
            "late-night-only":     1.0,
            "steady-allrounder":   1.4,
            "struggling-newbie":   0.6,
        }
        # Merchant weights.
        mer_weight = {
            "rising-star":         1.8,
            "lunch-dominant":      1.4,
            "dinner-dominant":     1.4,
            "weekend-spike":       1.2,
            "declining-weekends":  1.0,
            "balanced":            1.0,
        }

        # ---- Pre-compute dietary-compatible merchants per customer ----
        # For customers with dietary_prefs, restrict the merchant pool to
        # merchants that offer at least one menu item matching any of their
        # prefs. With no prefs, all merchants are compatible. Soft constraint:
        # 90% of orders go to compatible merchants, 10% to anyone (people
        # occasionally order things outside their usual diet).
        def _merchant_compatible(m: M.Merchant, prefs: list[str]) -> bool:
            if not prefs:
                return True
            for item in m.menu_items:
                if any(p in (item.tags or []) for p in prefs):
                    return True
            return False

        compatible_merchants: dict[int, list[M.Merchant]] = {}
        for c in customers:
            prefs = c.dietary_prefs or []
            compat = [m for m in merchants if _merchant_compatible(m, prefs)]
            # Edge case: no merchants offer the diet → use all merchants
            compatible_merchants[c.id] = compat or merchants
        diet_aligned_count = sum(1 for c in customers if c.dietary_prefs)
        print(f"  ✓ Pre-computed dietary-compatible merchant pools for {diet_aligned_count} customers with prefs")

        orders_created = 0
        cancelled = 0
        for _ in range(TARGET_ORDERS):
            customer = random.choices(customers, weights=c_weights, k=1)[0]
            cp = customer.behavior_persona

            # Apply dietary alignment 90% of the time (10% lets reality leak in).
            cust_compat = compatible_merchants.get(customer.id, merchants)
            base_pool = cust_compat if random.random() < 0.90 else merchants

            # Pick merchant biased by city + persona, scoped to the dietary-aligned pool.
            same_city_merchants = [m for m in base_pool if m.city_id == customer.city_id]
            pool_m = same_city_merchants if (same_city_merchants and random.random() < 0.85) else base_pool
            if not pool_m:
                pool_m = base_pool  # final safety net
            m_w = [mer_weight.get(m.behavior_persona, 1.0) for m in pool_m]
            merchant = random.choices(pool_m, weights=m_w, k=1)[0]

            # Pick driver biased by city + persona.
            same_city_drivers = [d for d in drivers if d.city_id == merchant.city_id]
            pool_d = same_city_drivers if same_city_drivers else drivers
            d_w = [drv_weight.get(d.behavior_persona, 1.0) for d in pool_d]
            driver = random.choices(pool_d, weights=d_w, k=1)[0]

            city = next(c for c in cities if c.id == merchant.city_id)

            # Total: high-spender skews up.
            if cp == "high-spender":
                total = round(random.uniform(35.0, 120.0), 2)
            elif cp == "new-user":
                total = round(random.uniform(8.0, 35.0), 2)
            else:
                total = round(random.uniform(8.0, 65.0), 2)

            # Cancellation tied to driver persona.
            cancel_p = driver.cancel_rate
            if random.random() < cancel_p:
                status = "cancelled"
                cancelled += 1
            else:
                status = "completed"

            # Driver earning skews higher for low-cancel veterans (incentives stack).
            base_take = random.uniform(0.65, 0.80)
            if driver.behavior_persona == "low-cancel-veteran":
                base_take = random.uniform(0.72, 0.85)
            driver_earning = round(total * base_take, 2)

            ts = _customer_timestamp(cp)

            # Merchant-persona override of timestamp.
            mp = merchant.behavior_persona
            if mp == "weekend-spike" and ts.weekday() < 4 and random.random() < 0.55:
                shift = (5 - ts.weekday()) % 7
                ts = ts + timedelta(days=shift)
            elif mp == "declining-weekends" and ts.weekday() >= 5:
                # 30% drop in recent 14d weekend orders → re-roll some onto weekdays.
                if (datetime.utcnow() - ts).days <= 14 and random.random() < 0.55:
                    ts = ts - timedelta(days=2)
            elif mp == "rising-star":
                # Skew to recent — half chance to pull into last 30 days.
                if random.random() < 0.55:
                    days_ago = random.randint(0, 29)
                    ts = (datetime.utcnow() - timedelta(days=days_ago)).replace(
                        hour=ts.hour, minute=ts.minute, second=0, microsecond=0
                    )
            elif mp == "lunch-dominant" and random.random() < 0.6:
                ts = ts.replace(hour=random.choice([11, 12, 13]))
            elif mp == "dinner-dominant" and random.random() < 0.6:
                ts = ts.replace(hour=random.choice([18, 19, 20, 21]))

            # Driver-persona timestamp influence.
            dp = driver.behavior_persona
            if dp == "weekend-warrior" and ts.weekday() < 4 and random.random() < 0.5:
                shift = (5 - ts.weekday()) % 7
                ts = ts + timedelta(days=shift)
            elif dp == "lunch-rush" and random.random() < 0.5:
                ts = ts.replace(hour=random.choice([11, 12, 13]))
            elif dp == "late-night-only" and random.random() < 0.6:
                ts = ts.replace(hour=random.choice([22, 23, 0, 1, 2]))

            # Pick fine-grained cells for pickup + dropoff.
            # Pickup cell = a cell within the merchant's zone, weighted by cell_type.
            pickup_candidates = cells_by_city_zone.get((city.id, merchant.zone), [])
            if pickup_candidates:
                pw = [CELL_TYPE_ORDER_WEIGHT.get(c.cell_type, 1.0) for c in pickup_candidates]
                pickup_cell = random.choices(pickup_candidates, weights=pw, k=1)[0]
                pickup_cell_id = pickup_cell.id
            else:
                pickup_cell_id = None
            # Dropoff cell = a cell within the random dropoff zone (weighted)
            dropoff_zone = _weighted_zone(city.zones)
            dropoff_candidates = cells_by_city_zone.get((city.id, dropoff_zone), [])
            if dropoff_candidates:
                dw = [CELL_TYPE_ORDER_WEIGHT.get(c.cell_type, 1.0) for c in dropoff_candidates]
                dropoff_cell_id = random.choices(dropoff_candidates, weights=dw, k=1)[0].id
            else:
                dropoff_cell_id = None

            o = M.Order(
                customer_id=customer.id,
                merchant_id=merchant.id,
                driver_id=driver.id,
                city_id=city.id,
                pickup_zone=merchant.zone,
                dropoff_zone=dropoff_zone,
                pickup_cell_id=pickup_cell_id,
                dropoff_cell_id=dropoff_cell_id,
                total=total,
                driver_earning=driver_earning if status == "completed" else 0.0,
                status=status,
                created_at=ts,
            )
            s.add(o); orders_created += 1
        print(f"  ✓ {orders_created} orders ({cancelled} cancelled) — cells assigned")
        s.flush()

        # ------- Precompute risk_score / risk_decision for every order -------
        # Strategy: point-in-time. For each customer, sort their orders chronologically
        # and use a sliding 7-day window for recent_orders / recent_cancels relative
        # to *that order's* created_at. avg_order_value uses all-orders mean (matches
        # the live tool, which queries all orders without a time filter).
        cust_orders: dict[int, list[M.Order]] = defaultdict(list)
        all_orders_rows = s.execute(select(M.Order)).scalars().all()
        for o in all_orders_rows:
            cust_orders[o.customer_id].append(o)

        cust_signup: dict[int, datetime] = {c.id: c.signup_date for c in customers}

        decisions = {"approve": 0, "review": 0, "block": 0}
        for cid, orders in cust_orders.items():
            orders.sort(key=lambda x: x.created_at)
            avg_total = (sum(o.total for o in orders) / len(orders)) if orders else 0.0
            signup = cust_signup.get(cid)
            # Sliding 7d window
            left = 0
            for i, o in enumerate(orders):
                window_start = o.created_at - timedelta(days=7)
                while left < i and orders[left].created_at < window_start:
                    left += 1
                window = orders[left:i + 1]
                recent_n = len(window)
                recent_cancels = sum(1 for w in window if w.status == "cancelled")
                tenure_days = (o.created_at - signup).days if signup else 0

                anomaly_score, flags = compute_customer_anomaly_score(
                    tenure_days=tenure_days,
                    recent_orders_7d=recent_n,
                    recent_cancels_7d=recent_cancels,
                )
                late_night = o.created_at.hour >= 22 or o.created_at.hour < 5
                risk, decision, _ = compute_order_risk_score(
                    anomaly_score=anomaly_score,
                    anomaly_flags=flags or ["no anomalies detected"],
                    avg_order_value=avg_total,
                    estimated_total=o.total,
                    late_night=late_night,
                )
                o.risk_score = risk
                o.risk_decision = decision
                decisions[decision] = decisions.get(decision, 0) + 1
        print(f"📊 risk distribution: approve={decisions['approve']}, review={decisions['review']}, block={decisions['block']}")

        # ------- Incentives -------
        now = datetime.utcnow()
        incentive_templates = [
            ("Weekend Surge Bonus", "Earn extra on every completed trip Fri-Sun evenings.", 8.0),
            ("Lunch Rush Reward", "Bonus per completed delivery between 11am-2pm.", 4.0),
            ("Late-Night Hero", "Top up for late-night trips (10pm-2am).", 6.0),
            ("Zone Hotspot", "Extra payout for trips originating in this zone.", 5.0),
            ("Streak Bonus", "Complete 10 trips in a day for a flat bonus.", 25.0),
            ("New Driver Boost", "Higher payout for drivers in their first 60 days.", 10.0),
        ]
        for _ in range(10):
            city = random.choice(cities)
            title, desc, bonus = random.choice(incentive_templates)
            zone = _weighted_zone(city.zones) if "Zone" in title else None
            inc = M.Incentive(
                city_id=city.id,
                title=title,
                description=desc,
                vehicle_type=random.choice(["bike", "car", "any"]),
                zone=zone,
                bonus_amount=bonus,
                starts_at=now - timedelta(days=random.randint(1, 7)),
                ends_at=now + timedelta(days=random.randint(3, 21)),
            )
            s.add(inc)
        print("  ✓ 10 incentives")

        # ------- Cell demand aggregate (cell × hour × day_of_week) -------
        # Pre-aggregating here means the optimizer reads <600 indexed rows
        # per call instead of scanning 12k orders every time.
        agg_bucket = defaultdict(
            lambda: {"orders": 0, "earnings": 0.0, "drivers": set()}
        )
        all_orders_for_agg = s.execute(
            select(M.Order).where(M.Order.status == "completed", M.Order.pickup_cell_id.is_not(None))
        ).scalars().all()
        for o in all_orders_for_agg:
            key = (o.pickup_cell_id, o.created_at.hour, o.created_at.weekday())
            agg_bucket[key]["orders"] += 1
            agg_bucket[key]["earnings"] += o.driver_earning or 0
            agg_bucket[key]["drivers"].add(o.driver_id)

        dem_rows = 0
        for (cell_id, hour, dow), v in agg_bucket.items():
            n = v["orders"]
            s.add(M.CellDemandHourly(
                cell_id=cell_id,
                hour=hour,
                day_of_week=dow,
                orders_total=n,
                drivers_unique=len(v["drivers"]),
                earnings_total=round(v["earnings"], 2),
                avg_earning_per_trip=round(v["earnings"] / max(n, 1), 2),
            ))
            dem_rows += 1
        s.flush()
        print(f"  ✓ {dem_rows} cell × hour × dow demand rows")

        # ------- Cell travel matrix (all within-city pairs) -------
        # Off-peak base minutes from Haversine distance × city avg speed.
        # Optimizer applies hour-of-day traffic multiplier at runtime.
        CITY_AVG_KMH = {
            "Singapore":     30.0,
            "Jakarta":       22.0,
            "Bangkok":       24.0,
            "Manila":        20.0,
            "Kuala Lumpur":  28.0,
        }
        travel_rows = 0
        for city in cities:
            cells = cells_by_city.get(city.id, [])
            avg_speed = CITY_AVG_KMH.get(city.name, 25.0)
            for c1 in cells:
                for c2 in cells:
                    dist = _haversine_km(c1.center_lat, c1.center_lon, c2.center_lat, c2.center_lon)
                    base_min = (dist / avg_speed) * 60.0
                    s.add(M.CellTravelMinutes(
                        from_cell_id=c1.id,
                        to_cell_id=c2.id,
                        base_minutes=round(base_min, 2),
                        distance_km=round(dist, 3),
                    ))
                    travel_rows += 1
            s.flush()
        print(f"  ✓ {travel_rows} cell-travel pairs (all-pairs intra-city)")

        # ------- Auth users — full identity rows with email/phone/avatar -------
        # Hash the shared default password ONCE (bcrypt is slow). Demo logins keep
        # the simple "password" so the documented credentials still work; users
        # registered via /api/auth/register will set their own.
        shared_hash = hash_password(DEFAULT_PASSWORD)
        now_utc = datetime.utcnow()

        auth_count = 0
        used_emails: set[str] = set()

        def _unique_email(name: str, role: str, idx: int) -> str:
            """Return a unique email — append a suffix if collision (rare)."""
            base = _email_for(name, role, idx)
            if base not in used_emails:
                used_emails.add(base); return base
            n = 2
            while True:
                cand = base.replace("@", f".{n}@")
                if cand not in used_emails:
                    used_emails.add(cand); return cand
                n += 1

        for i, d in enumerate(drivers, start=1):
            city_name = d.city.name
            email = _unique_email(d.name, "driver", i)
            s.add(M.AuthUser(
                email=email,
                username=f"driver{i}",
                phone=d.phone,
                password_hash=shared_hash,
                full_name=d.name,
                avatar_url=_avatar_url(d.name),
                role="driver",
                driver_id=d.id,
                is_active=True,
                is_verified=True,
                created_at=d.joined_date,
                updated_at=now_utc,
            )); auth_count += 1

        for i, c in enumerate(customers, start=1):
            email = _unique_email(c.name, "customer", i)
            s.add(M.AuthUser(
                email=email,
                username=f"customer{i}",
                phone=_phone_for(c.city.name),
                password_hash=shared_hash,
                full_name=c.name,
                avatar_url=_avatar_url(c.name),
                role="customer",
                customer_id=c.id,
                is_active=True,
                is_verified=True,
                created_at=c.signup_date,
                updated_at=now_utc,
            )); auth_count += 1

        for i, mrow in enumerate(merchants, start=1):
            email = _unique_email(mrow.name, "merchant", i)
            s.add(M.AuthUser(
                email=email,
                username=f"merchant{i}",
                phone=_phone_for(mrow.city.name),
                password_hash=shared_hash,
                full_name=mrow.name,
                avatar_url=_avatar_url(mrow.name),
                role="merchant",
                merchant_id=mrow.id,
                is_active=True,
                is_verified=True,
                created_at=now_utc - timedelta(days=random.randint(60, 400)),
                updated_at=now_utc,
            )); auth_count += 1

        # ------- Admins (2) -------
        s.add(M.AuthUser(
            email="admin@grabwise.demo",
            username="admin",
            phone="+6580000001",
            password_hash=hash_password("admin"),
            full_name="GrabWise Admin",
            avatar_url=_avatar_url("Admin"),
            role="admin",
            is_active=True,
            is_verified=True,
            created_at=now_utc,
            updated_at=now_utc,
        )); auth_count += 1
        s.add(M.AuthUser(
            email="grabops@grabwise.demo",
            username="grabops",
            phone="+6580000002",
            password_hash=shared_hash,
            full_name="Grab Operations",
            avatar_url=_avatar_url("Grab Ops"),
            role="admin",
            is_active=True,
            is_verified=True,
            created_at=now_utc,
            updated_at=now_utc,
        )); auth_count += 1
        print(f"  ✓ {auth_count} auth users")

    print("✅ Seed complete.")
    print("📋 Demo logins:")
    print("   admin / admin")
    print("   driver1   / password   (and driver2 … driver200)")
    print("   customer1 / password   (and customer2 … customer50)")
    print("   merchant1 / password   (and merchant2 … merchant50)")
    print("   Tip: you can also log in with the user's email (printed in /api/users)")


if __name__ == "__main__":
    reset_and_seed()
