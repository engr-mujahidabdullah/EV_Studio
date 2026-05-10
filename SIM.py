"""
EV Design Studio Pro
====================
A comprehensive Streamlit application for electric vehicle engineering,
powertrain sizing, battery analysis, performance simulation, and economics.

Run with:
    streamlit run SIM.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ============================================================
# CONSTANTS
# ============================================================
G = 9.81          # m/s^2
RHO_AIR = 1.225   # kg/m^3 (air density at sea level)

# ============================================================
# VEHICLE PRESETS
# ============================================================
PRESETS = {
    "3W Rickshaw (Urban EV)": {
        "m": 450, "cd": 0.65, "area": 1.0, "cr": 0.015,
        "v_nom": 48, "ah": 90, "eta_motor": 0.90, "eta_regen": 0.65,
        "v_cell": 3.2, "ah_cell": 100.0, "wheel_radius": 0.28,
        "gear_ratio": 8.5, "peak_torque": 60, "battery_thermal_mass": 15.0,
        "description": "Typical Indian 3-wheeler EV replacement for CNG rickshaws",
    },
    "2W Scooter (Compact)": {
        "m": 160, "cd": 0.60, "area": 0.5, "cr": 0.015,
        "v_nom": 48, "ah": 35, "eta_motor": 0.92, "eta_regen": 0.70,
        "v_cell": 3.7, "ah_cell": 3.0, "wheel_radius": 0.26,
        "gear_ratio": 7.0, "peak_torque": 30, "battery_thermal_mass": 5.0,
        "description": "Compact urban electric scooter for last-mile mobility",
    },
    "Light EV Car (City)": {
        "m": 1200, "cd": 0.28, "area": 2.0, "cr": 0.012,
        "v_nom": 400, "ah": 60, "eta_motor": 0.94, "eta_regen": 0.75,
        "v_cell": 3.7, "ah_cell": 50.0, "wheel_radius": 0.32,
        "gear_ratio": 9.0, "peak_torque": 200, "battery_thermal_mass": 50.0,
        "description": "Small electric city car (Tata Tiago-class)",
    },
    "Performance EV (Sport)": {
        "m": 1800, "cd": 0.22, "area": 2.2, "cr": 0.010,
        "v_nom": 400, "ah": 100, "eta_motor": 0.96, "eta_regen": 0.80,
        "v_cell": 3.7, "ah_cell": 50.0, "wheel_radius": 0.34,
        "gear_ratio": 9.5, "peak_torque": 500, "battery_thermal_mass": 80.0,
        "description": "High-performance electric sports car",
    },
}

# ============================================================
# CORE PHYSICS ENGINE
# ============================================================

def tractive_forces(v_kmh, p, gradient_pct=0.0):
    """Returns (total, f_aero, f_roll, f_grade) in Newtons."""
    v = v_kmh / 3.6
    theta = np.arctan(gradient_pct / 100.0)
    f_aero  = 0.5 * RHO_AIR * p["cd"] * p["area"] * v ** 2
    f_roll  = p["m"] * G * p["cr"] * np.cos(theta)
    f_grade = p["m"] * G * np.sin(theta)
    return f_aero + f_roll + f_grade, f_aero, f_roll, f_grade


def motor_power_kw(v_kmh, p, gradient_pct=0.0):
    """Returns (motor input kW, wheel kW)."""
    total, *_ = tractive_forces(v_kmh, p, gradient_pct)
    v_ms = v_kmh / 3.6
    p_wheel_kw = total * v_ms / 1000.0
    p_motor_kw = p_wheel_kw / p["eta_motor"] if p_wheel_kw > 0 else p_wheel_kw
    return p_motor_kw, p_wheel_kw


def simulate_acceleration(p, peak_power_kw, target_kmh=80.0):
    """Simulates 0 to target_kmh acceleration under peak power constraint."""
    dt = 0.05
    v  = 0.01   # m/s
    t  = 0.0
    t_log, v_log = [0.0], [0.0]
    f_max_torque = p["peak_torque"] * p["gear_ratio"] / p["wheel_radius"]
    target_ms = target_kmh / 3.6

    while v < target_ms and t < 120:
        _, f_aero, f_roll, _ = tractive_forces(v * 3.6, p)
        f_avail = min((peak_power_kw * 1000) / v, f_max_torque)
        f_net   = max(0, f_avail - f_aero - f_roll)
        a = f_net / p["m"]
        v += a * dt
        t += dt
        v_log.append(min(v * 3.6, target_kmh))
        t_log.append(t)

    return np.array(t_log), np.array(v_log)


def simulate_drive_cycle(v_kmh_profile, p, dt=1.0):
    """
    Simulate energy use over a velocity profile (km/h array) with regen braking.
    Returns (energy_Wh, distance_km, cumulative_energy_Wh, cumulative_dist_km).
    """
    v     = v_kmh_profile / 3.6
    accel = np.concatenate(([0.0], np.diff(v) / dt))

    f_aero = 0.5 * RHO_AIR * p["cd"] * p["area"] * v ** 2
    f_roll = p["m"] * G * p["cr"] * np.ones_like(v)
    f_acc  = p["m"] * accel

    p_wheel_prop = (f_aero + f_roll + np.maximum(f_acc, 0)) * v
    p_motor_in   = np.where(p_wheel_prop > 0, p_wheel_prop / p["eta_motor"], 0.0)

    p_regen_elec = np.maximum(-f_acc * v, 0.0) * p["eta_regen"] * p["eta_motor"]

    net_power    = p_motor_in - p_regen_elec
    cumul_energy = np.cumsum(net_power * dt / 3600.0)
    cumul_dist   = np.cumsum(v * dt / 1000.0)
    return float(cumul_energy[-1]), float(cumul_dist[-1]), cumul_energy, cumul_dist


def battery_soh_aging(cycles, temp_c=25.0, dod=0.80):
    """Returns (cycle_array, soh%, r_growth, eol_cycle)."""
    c          = np.arange(0, cycles, 10)
    temp_stress = np.exp(0.02 * (temp_c - 25) / 10)
    dod_stress  = 1.0 + 0.5 * (dod - 0.8)
    k           = temp_stress * dod_stress
    soh         = np.clip(100 - (0.005 * k * c + 1.5 * np.exp(c / (1500 / k))), 0, 100)
    r_growth    = 1.0 + (100 - soh) * 0.022 * temp_stress
    eol_idx     = np.where(soh <= 80)[0]
    eol_cycle   = int(c[eol_idx[0]]) if len(eol_idx) > 0 else int(cycles)
    return c, soh, r_growth, eol_cycle


def motor_efficiency_map(motor_type):
    """Generate a realistic 2-D efficiency map (RPM x Torque)."""
    rpm    = np.linspace(100, 6000, 60)
    torque = np.linspace(1,   100,  60)
    R, T   = np.meshgrid(rpm, torque)
    cfg = {
        "PMSM":           (0.96, 3500, 55, 0.08, 0.10),
        "Induction (IM)": (0.91, 2500, 40, 0.12, 0.14),
        "BLDC":           (0.93, 3000, 50, 0.09, 0.12),
    }
    pk, r0, t0, kr, kt = cfg.get(motor_type, cfg["PMSM"])
    eff = pk - kr * ((R - r0) / r0) ** 2 - kt * ((T - t0) / t0) ** 2
    return R, T, np.clip(eff, 0.50, pk)


def battery_thermal_model(p, current_A, dt=1.0, t_ambient=25.0):
    """Lumped-capacitance thermal model. Returns temperature array (Celsius)."""
    R_int     = 0.05
    C_thermal = p["battery_thermal_mass"] * 900.0
    R_thermal = 0.8
    T, temps  = t_ambient, []
    for I in current_A:
        Q_gen = I ** 2 * R_int
        Q_cool = (T - t_ambient) / R_thermal
        T = T + (Q_gen - Q_cool) * dt / C_thermal
        temps.append(T)
    return np.array(temps)


def tco_analysis(p, cfg):
    """Total Cost of Ownership over lifetime. Returns dict of results."""
    yr       = np.arange(1, cfg["years"] + 1)
    kwh      = (p["v_nom"] * p["ah"]) / 1000.0
    bat_cost = kwh * cfg["bat_cost_kwh"]

    ev_annual  = (cfg["annual_km"] * cfg["wh_km"] / 1000) * cfg["elec_price"]
    ev_maint   = cfg["ev_price"] * 0.01
    ev_bat_rep = bat_cost if cfg["years"] > 8 else 0
    ev_cum     = cfg["ev_price"] + (ev_annual + ev_maint) * yr
    ev_cum[-1] += ev_bat_rep

    ice_fuel  = (cfg["annual_km"] * cfg["l_100km"] / 100) * cfg["fuel_price"]
    ice_maint = cfg["ice_price"] * 0.025
    ice_cum   = cfg["ice_price"] + (ice_fuel + ice_maint) * yr

    breakeven = next((int(y) for y, ev, ic in zip(yr, ev_cum, ice_cum) if ev <= ic), None)
    return {
        "yr": yr, "ev_cum": ev_cum, "ice_cum": ice_cum,
        "ev_total": ev_cum[-1], "ice_total": ice_cum[-1],
        "savings": ice_cum[-1] - ev_cum[-1],
        "breakeven": breakeven,
        "kwh": kwh, "bat_cost": bat_cost,
    }


def generate_drive_cycle(name, duration=1200):
    """Return (t, v_kmh) arrays for standard drive cycles."""
    t = np.arange(0, duration, 1)
    np.random.seed(0)
    if name == "Urban Stop-Go":
        v = np.zeros(len(t))
        for i, ti in enumerate(t):
            ph = ti % 120
            if ph < 30:   v[i] = (ph / 30) * 35
            elif ph < 60: v[i] = 35
            elif ph < 80: v[i] = 35 - ((ph - 60) / 20) * 35
    elif name == "Highway Cruise":
        v = np.where(t < 60, t * (110 / 60), 110.0)
    elif name == "Mixed Duty":
        v = 40 + 25 * np.sin(2 * np.pi * t / 300) + 5 * np.random.randn(len(t))
        v = np.clip(v, 0, 100)
    else:   # WLTP-style
        v = 30 + 20 * np.sin(2 * np.pi * t / 200) + 10 * np.sin(2 * np.pi * t / 80)
        v = np.clip(v, 0, 130)
    return t, np.clip(v, 0, None)


def _shape_dimensions(area_m2, shape):
    """Return width/height that preserve frontal area with shape-specific aspect."""
    aspect_map = {
        "Sedan": 2.2,
        "SUV": 1.8,
        "Coupe": 2.5,
        "Hatchback": 2.0,
        "Van": 1.6,
    }
    area_eff = max(float(area_m2), 0.2)
    aspect = aspect_map.get(shape, 2.1)
    width = np.sqrt(area_eff * aspect)
    height = area_eff / max(width, 0.1)
    return width, height


def _shape_half_height(x_local, width, height, shape):
    """Half-height profile y(x) for a simple automotive side silhouette."""
    xn = np.clip(x_local / (0.5 * width), -1.0, 1.0)
    root = np.sqrt(np.clip(1.0 - xn ** 2, 0.0, None))

    if shape == "SUV":
        y_scale = np.clip(0.90 - 0.12 * np.maximum(xn, 0.0), 0.55, 1.0)
        y_half = 0.5 * height * y_scale
    elif shape == "Coupe":
        y_half = 0.5 * height * (0.82 * root ** 0.70)
    elif shape == "Hatchback":
        front = 0.90 * np.sqrt(np.clip(1.0 - ((xn + 0.12) / 1.08) ** 2, 0.0, None))
        rear = np.clip(0.92 - 0.52 * np.maximum(xn, 0.0), 0.36, 0.92)
        y_half = 0.5 * height * np.minimum(front, rear)
    elif shape == "Van":
        y_half = 0.5 * height * np.clip(0.98 - 0.08 * np.maximum(xn, 0.0), 0.72, 1.0)
    else:  # Sedan
        y_half = 0.5 * height * (0.90 * root ** 0.58 * (0.98 - 0.20 * np.maximum(xn, 0.0)))

    return np.clip(y_half, 0.08 * height, None)


def estimate_drag_from_flow(U, X, Y, area_m2, speed_ms, width):
    """Estimate drag from wake momentum deficit at a downstream control plane."""
    x_plane = 1.8 * width
    col = int(np.argmin(np.abs(X[0] - x_plane)))
    u_col = np.nan_to_num(U[:, col], nan=0.0)
    y_col = Y[:, col]
    dy = float(np.mean(np.diff(y_col))) if len(y_col) > 1 else 1.0
    depth = max(area_m2 / max(np.max(np.abs(y_col)) * 2, 0.1), 0.2)
    deficit = np.clip(speed_ms - u_col, 0.0, None)
    drag_n = RHO_AIR * depth * np.sum(u_col * deficit) * dy
    q = 0.5 * RHO_AIR * speed_ms ** 2
    cd_flow = drag_n / max(q * area_m2, 1e-6)
    return max(float(drag_n), 0.0), max(float(cd_flow), 0.0)


def aerodynamic_flow_field(area_m2, speed_kmh, shape="Sedan", nx=140, ny=70):
    """2D pseudo-flow field around an automotive body and drag estimate."""
    area_eff = max(float(area_m2), 0.2)
    speed_ms = max(float(speed_kmh) / 3.6, 0.2)
    width, height = _shape_dimensions(area_eff, shape)

    x = np.linspace(-3.0 * width, 5.0 * width, nx)
    y = np.linspace(-2.5 * height, 2.5 * height, ny)
    X, Y = np.meshgrid(x, y)

    U = np.full_like(X, speed_ms)
    V = np.zeros_like(X)

    shape_gain = {
        "Sedan": (0.40, 0.52, 0.50),
        "SUV": (0.46, 0.65, 0.55),
        "Coupe": (0.35, 0.43, 0.42),
        "Hatchback": (0.43, 0.59, 0.53),
        "Van": (0.48, 0.72, 0.56),
    }
    front_gain, wake_gain, deflect_gain = shape_gain.get(shape, shape_gain["Sedan"])

    sigma = 0.45 * width
    front = np.exp(-((X + 0.65 * width) ** 2) / (2 * sigma ** 2) - (Y ** 2) / (2 * (0.60 * height) ** 2))
    U *= (1 - front_gain * front)

    # Deflect streamlines away from body centerline near the nose.
    V += deflect_gain * speed_ms * (Y / max(height, 0.1)) * front

    wake = np.exp(-((X - 0.90 * width) ** 2) / (2 * (2.0 * width) ** 2) - (Y ** 2) / (2 * (0.74 * height) ** 2))
    U *= (1 - wake_gain * wake)

    y_half = _shape_half_height(X, width, height, shape)
    core_body = (np.abs(X) <= (0.5 * width)) & (np.abs(Y) <= y_half)
    U = np.where(core_body, np.nan, np.clip(U, 0.0, None))
    V = np.where(core_body, np.nan, V)

    speed_map = np.sqrt(np.nan_to_num(U, nan=0.0) ** 2 + np.nan_to_num(V, nan=0.0) ** 2)
    drag_n, cd_flow = estimate_drag_from_flow(U, X, Y, area_eff, speed_ms, width)
    return X, Y, U, V, speed_map, width, height, drag_n, cd_flow


# ============================================================
# PLOTTING HELPERS
# ============================================================
DARK_BG  = "#0E1117"
PANEL_BG = "#1A1E2A"
GRID_CLR = "#2A2D3A"
TICK_CLR = "#CCCCCC"
C = {
    "primary":   "#00D4FF",
    "secondary": "#FF6B35",
    "success":   "#00FF88",
    "warning":   "#FFD700",
    "danger":    "#FF4444",
}


def _apply_dark(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TICK_CLR)
    ax.xaxis.label.set_color(TICK_CLR)
    ax.yaxis.label.set_color(TICK_CLR)
    ax.title.set_color("white")
    for sp in ax.spines.values():
        sp.set_color(GRID_CLR)
    ax.grid(True, alpha=0.2, color=GRID_CLR)


def new_fig(figsize=(8, 4)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=DARK_BG)
    _apply_dark(ax)
    return fig, ax


def draw_architecture_diagram(vehicle_class, topology):
    """Draw simplified power-flow schematic for HEV/PHEV/EREV topologies."""
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def node(x, y, label, fc="#223048", ec="#4A6A9A", w=1.35, h=0.62):
        box = plt.Rectangle((x - w / 2, y - h / 2), w, h,
                            facecolor=fc, edgecolor=ec, linewidth=1.2)
        ax.add_patch(box)
        ax.text(x, y, label, color="white", ha="center", va="center", fontsize=8)

    def link(a, b, color=C["primary"], style="-", lw=1.8):
        ax.annotate("", xy=b, xytext=a,
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle=style))

    if topology == "Series":
        node(1.0, 3.8, "Fuel Tank", fc="#3C2A2A")
        node(2.5, 3.8, "ICE", fc="#4A3131")
        node(4.0, 3.8, "Generator", fc="#2F4A3F")
        node(5.5, 2.5, "DC Bus", fc="#2E3D5A")
        node(4.0, 1.2, "Battery", fc="#304A2F")
        node(7.0, 2.5, "Motor", fc="#2B4D63")
        node(8.7, 2.5, "Wheels", fc="#3A3A3A")
        if vehicle_class in ["PHEV", "EREV"]:
            node(1.8, 1.2, "Charge Port", fc="#2B5A45")
            link((2.5, 1.2), (3.3, 1.2), color=C["success"])

        link((1.7, 3.8), (1.85, 3.8), color=C["secondary"])
        link((3.15, 3.8), (3.35, 3.8), color=C["secondary"])
        link((4.65, 3.8), (4.85, 2.8), color=C["primary"])
        link((4.65, 1.2), (4.85, 2.2), color=C["success"])
        link((6.15, 2.5), (6.3, 2.5), color=C["primary"])
        link((7.65, 2.5), (8.0, 2.5), color=C["primary"])
        link((8.05, 2.2), (6.9, 2.2), color=C["warning"], style="--", lw=1.2)
        link((6.45, 2.2), (4.55, 1.35), color=C["warning"], style="--", lw=1.2)

    elif topology == "Parallel":
        node(1.0, 3.8, "Fuel Tank", fc="#3C2A2A")
        node(2.5, 3.8, "ICE", fc="#4A3131")
        node(4.3, 3.8, "Transmission", fc="#3D365A")
        node(6.0, 3.8, "Final Drive", fc="#3D365A")
        node(8.0, 3.8, "Wheels", fc="#3A3A3A")

        node(2.5, 1.3, "Battery", fc="#304A2F")
        node(4.3, 1.3, "Inverter", fc="#2E3D5A")
        node(6.0, 1.3, "E-Motor", fc="#2B4D63")
        if vehicle_class == "PHEV":
            node(0.9, 1.3, "Charge Port", fc="#2B5A45")
            link((1.55, 1.3), (1.85, 1.3), color=C["success"])

        link((1.7, 3.8), (1.85, 3.8), color=C["secondary"])
        link((3.15, 3.8), (3.6, 3.8), color=C["secondary"])
        link((5.0, 3.8), (5.3, 3.8), color=C["secondary"])
        link((6.7, 3.8), (7.3, 3.8), color=C["secondary"])

        link((3.15, 1.3), (3.6, 1.3), color=C["success"])
        link((5.0, 1.3), (5.3, 1.3), color=C["success"])
        link((6.6, 1.6), (6.6, 3.2), color=C["primary"])
        link((7.4, 3.5), (6.3, 1.6), color=C["warning"], style="--", lw=1.2)
        link((5.7, 1.1), (3.1, 1.1), color=C["warning"], style="--", lw=1.2)

    else:  # Series-Parallel
        node(1.0, 3.8, "Fuel Tank", fc="#3C2A2A")
        node(2.4, 3.8, "ICE", fc="#4A3131")
        node(4.0, 3.8, "Power Split", fc="#3D365A")
        node(5.7, 3.8, "Transmission", fc="#3D365A")
        node(7.7, 3.8, "Wheels", fc="#3A3A3A")

        node(4.0, 2.2, "Generator", fc="#2F4A3F")
        node(2.4, 1.0, "Battery", fc="#304A2F")
        node(4.0, 1.0, "Inverter", fc="#2E3D5A")
        node(5.7, 1.0, "E-Motor", fc="#2B4D63")
        if vehicle_class == "PHEV":
            node(0.9, 1.0, "Charge Port", fc="#2B5A45")
            link((1.55, 1.0), (1.75, 1.0), color=C["success"])

        link((1.7, 3.8), (1.75, 3.8), color=C["secondary"])
        link((3.1, 3.8), (3.35, 3.8), color=C["secondary"])
        link((4.65, 3.8), (5.0, 3.8), color=C["secondary"])
        link((6.35, 3.8), (7.0, 3.8), color=C["secondary"])

        link((4.0, 3.5), (4.0, 2.55), color=C["primary"])
        link((3.05, 1.0), (3.3, 1.0), color=C["success"])
        link((4.65, 1.0), (5.0, 1.0), color=C["success"])
        link((5.8, 1.35), (5.8, 3.2), color=C["primary"])
        link((7.0, 3.5), (5.9, 1.35), color=C["warning"], style="--", lw=1.2)
        link((5.4, 0.8), (3.0, 0.8), color=C["warning"], style="--", lw=1.2)

    ax.text(0.3, 0.25, "Solid arrows: propulsion power  |  Dashed arrows: regenerative path",
            color=TICK_CLR, fontsize=8, ha="left")
    ax.set_title(f"{vehicle_class} Architecture: {topology}", color="white", fontsize=11)
    return fig


def classify_hybridization(motor_kw, battery_kwh, ice_kw):
    """Classify degree of hybridization using power split and battery size."""
    motor = max(float(motor_kw), 0.0)
    battery = max(float(battery_kwh), 0.0)
    ice = max(float(ice_kw), 0.0)
    total_prop = max(motor + ice, 1e-6)

    doh = motor / total_prop
    ev_power_only = motor

    # Rule set for practical architecture grouping.
    if battery >= 8.0 and motor >= 25.0:
        hybrid_type = "Plug-in Hybrid"
        note = "Large battery and meaningful e-motor power enable substantial EV-only driving."
    elif doh >= 0.28 and battery >= 1.2 and motor >= 18.0:
        hybrid_type = "Full Hybrid"
        note = "Motor can independently propel the vehicle in limited low/medium load conditions."
    elif doh >= 0.08 and battery >= 0.3 and motor >= 6.0:
        hybrid_type = "Mild Hybrid"
        note = "Electric machine mainly assists ICE and supports regen/start-stop, with limited EV-only drive."
    else:
        hybrid_type = "Micro Hybrid / Start-Stop"
        note = "Low electrification, mainly idle-stop and recuperation support."

    return {
        "type": hybrid_type,
        "doh": doh,
        "doh_pct": doh * 100.0,
        "ev_power_only": ev_power_only,
        "total_prop": total_prop,
        "note": note,
    }


# ============================================================
# STREAMLIT APP
# ============================================================
st.set_page_config(
    page_title="EV Design Studio Pro",
    page_icon="ⓔ",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
div[data-testid="stMetricValue"] { font-size: 1.6rem; color: #00D4FF; }
.stTabs [data-baseweb="tab"]     { font-weight: 600; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align:center;color:#00D4FF;'>EV Design Studio Pro</h1>"
    "<p style='text-align:center;color:#AAAAAA;'>Comprehensive Electric Vehicle Engineering and Analytics Platform</p>",
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("## Vehicle Configuration")

preset_names  = list(PRESETS.keys()) + ["Custom"]
preset_choice = st.sidebar.selectbox("Vehicle Preset", preset_names)

if preset_choice != "Custom":
    p = PRESETS[preset_choice].copy()
    st.sidebar.caption(p["description"])
else:
    st.sidebar.markdown("#### Custom Parameters")
    p = {
        "m":                   st.sidebar.number_input("Mass (kg)", 50, 3000, 450),
        "cd":                  st.sidebar.slider("Drag Coefficient Cd", 0.15, 1.0, 0.65),
        "area":                st.sidebar.number_input("Frontal Area (m2)", 0.3, 4.0, 1.0),
        "cr":                  st.sidebar.slider("Rolling Resistance Cr", 0.005, 0.03, 0.015),
        "v_nom":               st.sidebar.number_input("Pack Voltage (V)", 24, 800, 48),
        "ah":                  st.sidebar.number_input("Pack Capacity (Ah)", 10, 500, 90),
        "eta_motor":           st.sidebar.slider("Motor Efficiency", 0.70, 0.98, 0.90),
        "eta_regen":           st.sidebar.slider("Regen Efficiency", 0.50, 0.85, 0.65),
        "v_cell": 3.2, "ah_cell": 100.0, "wheel_radius": 0.28,
        "gear_ratio": 8.5, "peak_torque": 60, "battery_thermal_mass": 15.0,
        "description": "Custom vehicle",
    }

pack_kwh = (p["v_nom"] * p["ah"]) / 1000.0

st.sidebar.markdown("---")
st.sidebar.markdown("### Live Range Estimator")
curr_soc    = st.sidebar.slider("Current SOC (%)", 0, 100, 80)
drive_style = st.sidebar.selectbox("Driving Style",
                                    ["Eco (35 Wh/km)", "Normal (45 Wh/km)", "Sport (65 Wh/km)"])
wh_km = {"Eco (35 Wh/km)": 35, "Normal (45 Wh/km)": 45, "Sport (65 Wh/km)": 65}[drive_style]
est_range = (curr_soc / 100.0) * pack_kwh * 1000.0 / wh_km

st.sidebar.metric("Pack Energy",      f"{pack_kwh:.2f} kWh")
st.sidebar.metric("Estimated Range",  f"{est_range:.1f} km")
st.sidebar.progress(curr_soc / 100)
st.sidebar.markdown("---")
st.sidebar.caption("EV Design Studio Pro - 2026 EV R&D Lab")

# ============================================================
# TOP KPI ROW
# ============================================================
p_motor_80, _ = motor_power_kw(80, p)
kp1, kp2, kp3, kp4, kp5 = st.columns(5)
kp1.metric("Pack Energy",         f"{pack_kwh:.2f} kWh")
kp2.metric("Vehicle Mass",        f"{p['m']} kg")
kp3.metric("Motor Power @80km/h", f"{p_motor_80:.1f} kW")
kp4.metric("Motor Efficiency",    f"{p['eta_motor']*100:.0f}%")
kp5.metric("Range Estimate",      f"{est_range:.0f} km")
st.markdown("---")

# ============================================================
# MAIN TABS
# ============================================================
(tab_powertrain,
 tab_battery,
 tab_performance,
 tab_drivecycle,
 tab_economics,
 tab_architecture,
 tab_analytics) = st.tabs([
    "Powertrain Sizing",
    "Battery Engineering",
    "Performance and Range",
    "Drive Cycle Analysis",
    "Economics and Environment",
    "EV Architectures",
    "Data Analytics",
])

# ==============================================================
# TAB 1: POWERTRAIN SIZING
# ==============================================================
with tab_powertrain:
    st.subheader("Powertrain Sizing and Motor Analysis")

    col_ctrl1, col_ctrl2 = st.columns(2)
    top_speed   = col_ctrl1.slider("Target Top Speed (km/h)", 20, 200, 80, key="ts_pw")
    gradient_pw = col_ctrl1.slider("Road Grade (%)", -5.0, 25.0, 0.0, key="grad_pw")
    motor_type  = col_ctrl2.selectbox("Motor Technology", ["PMSM", "Induction (IM)", "BLDC"])
    show_stack  = col_ctrl2.checkbox("Show Force Breakdown (stacked)", value=True)

    v_arr   = np.linspace(1, top_speed, 100)
    results = [tractive_forces(v, p, gradient_pw) for v in v_arr]
    f_tot   = np.array([r[0] for r in results])
    f_aero  = np.array([r[1] for r in results])
    f_roll  = np.array([r[2] for r in results])
    f_grade = np.array([r[3] for r in results])
    p_motor = np.array([motor_power_kw(v, p, gradient_pw)[0] for v in v_arr])
    p_wheel = np.array([motor_power_kw(v, p, gradient_pw)[1] for v in v_arr])

    pc1, pc2 = st.columns(2)
    with pc1:
        fig_f, ax_f = new_fig()
        if show_stack:
            ax_f.stackplot(v_arr, f_aero, f_roll, np.abs(f_grade),
                           labels=["Aerodynamic", "Rolling Resistance", "Grade"],
                           colors=[C["primary"], C["secondary"], C["warning"]], alpha=0.85)
            ax_f.legend(facecolor=PANEL_BG, labelcolor="white", fontsize=8)
        else:
            ax_f.plot(v_arr, f_tot, color=C["primary"], lw=2)
        ax_f.set_xlabel("Speed (km/h)"); ax_f.set_ylabel("Force (N)")
        ax_f.set_title("Tractive Effort Curve")
        st.pyplot(fig_f);  plt.close(fig_f)

    with pc2:
        fig_p, ax_p = new_fig()
        ax_p.plot(v_arr, p_motor, color=C["primary"],   lw=2.5, label="Motor Input Power")
        ax_p.plot(v_arr, p_wheel, color=C["success"],   lw=1.8, ls="--", label="Wheel Power")
        ax_p.fill_between(v_arr, p_wheel, p_motor, alpha=0.25, color=C["warning"],
                           label="Drivetrain Losses")
        ax_p.axhline(p_motor.max(), color=C["danger"], lw=1, ls=":",
                     label=f"Peak: {p_motor.max():.1f} kW")
        ax_p.set_xlabel("Speed (km/h)"); ax_p.set_ylabel("Power (kW)")
        ax_p.set_title("Power Requirement Profile")
        ax_p.legend(facecolor=PANEL_BG, labelcolor="white", fontsize=8)
        st.pyplot(fig_p);  plt.close(fig_p)

    st.markdown("#### Motor Efficiency Map")
    R_map, T_map, EFF_map = motor_efficiency_map(motor_type)

    mc1, mc2 = st.columns([2, 1])
    with mc1:
        fig_m, ax_m = plt.subplots(figsize=(8, 5), facecolor=DARK_BG)
        ax_m.set_facecolor(PANEL_BG)
        levels = np.arange(0.55, 0.97, 0.02)
        cp = ax_m.contourf(R_map, T_map, EFF_map, levels=levels, cmap="viridis")
        cs = ax_m.contour(R_map, T_map, EFF_map, levels=levels,
                           colors="white", alpha=0.3, linewidths=0.7)
        ax_m.clabel(cs, fmt="%.2f", fontsize=7, colors="white")
        cbar = fig_m.colorbar(cp, ax=ax_m)
        cbar.set_label("Efficiency", color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
        _apply_dark(ax_m)
        ax_m.set_xlabel("RPM"); ax_m.set_ylabel("Torque (Nm)")
        ax_m.set_title(f"{motor_type} Efficiency Map")
        st.pyplot(fig_m);  plt.close(fig_m)

    with mc2:
        st.markdown("**Motor Sizing Summary**")
        recommended_kw = np.ceil(p_motor.max() * 1.2)
        st.metric("Required Peak Power",  f"{p_motor.max():.1f} kW")
        st.metric("Recommended Motor",    f"{recommended_kw:.0f} kW (x1.2 margin)")
        st.metric("Max Tractive Force",   f"{f_tot.max():.0f} N")
        st.metric("Force @ 50 km/h",     f"{tractive_forces(50, p, gradient_pw)[0]:.0f} N")
        tech_notes = {
            "PMSM":           "Best-in-class efficiency. Rare-earth magnets. Ideal for cars.",
            "Induction (IM)": "Robust and low-cost. Slightly lower peak efficiency.",
            "BLDC":           "Simple controller, compact. Great for scooters and 3Ws.",
        }
        st.info(tech_notes[motor_type])

    st.markdown("#### 2D Aerodynamic Flow Field")
    af1, af2, af3, af4 = st.columns(4)
    flow_speed = af1.slider("Flow Speed (km/h)", 10, 180, int(top_speed), key="flow_speed")
    flow_area = af2.slider("Frontal Area for Flow (m2)", 0.3, 4.5, float(p["area"]), 0.05, key="flow_area")
    body_shape = af3.selectbox("Automotive Shape", ["Sedan", "SUV", "Coupe", "Hatchback", "Van"])
    show_quiver = af4.checkbox("Overlay velocity vectors", value=False)

    Xf, Yf, Uf, Vf, Sm, body_w, body_h, drag_flow_n, cd_flow = aerodynamic_flow_field(
        flow_area, flow_speed, shape=body_shape
    )

    fig_fl, ax_fl = plt.subplots(figsize=(9, 4.5), facecolor=DARK_BG)
    ax_fl.set_facecolor(PANEL_BG)
    cmap = plt.cm.plasma
    stream = ax_fl.streamplot(
        Xf, Yf, np.nan_to_num(Uf, nan=0.0), np.nan_to_num(Vf, nan=0.0),
        density=1.15, color=Sm, cmap=cmap, linewidth=1.0, arrowsize=0.9
    )
    if show_quiver:
        skip = (slice(None, None, 5), slice(None, None, 5))
        ax_fl.quiver(
            Xf[skip], Yf[skip], np.nan_to_num(Uf[skip], nan=0.0), np.nan_to_num(Vf[skip], nan=0.0),
            color="white", alpha=0.35, scale=250
        )
    x_prof = np.linspace(-0.5 * body_w, 0.5 * body_w, 250)
    y_prof = _shape_half_height(x_prof, body_w, body_h, body_shape)
    ax_fl.fill_between(x_prof, -y_prof, y_prof, color=C["secondary"], alpha=0.92, ec="white", lw=1.0)
    ax_fl.text(0, 0, body_shape, color="white", ha="center", va="center", fontsize=8)
    cbar = fig_fl.colorbar(stream.lines, ax=ax_fl)
    cbar.set_label("Local Flow Speed (m/s)", color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    _apply_dark(ax_fl)
    ax_fl.set_xlabel("X (m)")
    ax_fl.set_ylabel("Y (m)")
    ax_fl.set_title("Flow Pattern Around Vehicle Cross-Section")
    st.pyplot(fig_fl)
    plt.close(fig_fl)

    dm1, dm2, dm3 = st.columns(3)
    dm1.metric("Flow-Model Drag Force", f"{drag_flow_n:.1f} N")
    dm2.metric("Flow-Model Cd", f"{cd_flow:.3f}")
    dm3.metric("Classical Drag", f"{0.5 * RHO_AIR * p['cd'] * flow_area * (flow_speed / 3.6) ** 2:.1f} N")

    st.markdown("#### Area Sweep: Drag vs Frontal Area")
    sweep_areas = np.linspace(0.3, 4.5, 28)
    sweep_drag = []
    sweep_cd = []
    for ar in sweep_areas:
        _, _, _, _, _, _, _, d_n, cd_n = aerodynamic_flow_field(ar, flow_speed, shape=body_shape, nx=90, ny=46)
        sweep_drag.append(d_n)
        sweep_cd.append(cd_n)

    fig_sw, (ax_sw1, ax_sw2) = plt.subplots(2, 1, figsize=(9, 6), facecolor=DARK_BG)
    _apply_dark(ax_sw1)
    _apply_dark(ax_sw2)
    ax_sw1.plot(sweep_areas, sweep_drag, color=C["primary"], lw=2.2, label="Flow-model drag")
    ax_sw1.scatter([flow_area], [drag_flow_n], color=C["warning"], s=45, zorder=5, label="Current point")
    ax_sw1.set_ylabel("Drag Force (N)")
    ax_sw1.set_title(f"Drag Sweep at {flow_speed} km/h ({body_shape})")
    ax_sw1.legend(facecolor=PANEL_BG, labelcolor="white", fontsize=8)

    ax_sw2.plot(sweep_areas, sweep_cd, color=C["success"], lw=2.0, label="Flow-model Cd")
    ax_sw2.scatter([flow_area], [cd_flow], color=C["warning"], s=45, zorder=5)
    ax_sw2.set_xlabel("Frontal Area (m2)")
    ax_sw2.set_ylabel("Estimated Cd")
    st.pyplot(fig_sw)
    plt.close(fig_sw)

    ref_area = max(p["area"], 0.1)
    area_ratio = flow_area / ref_area
    st.caption(
        f"As frontal area increases, streamlines spread more and wake intensity grows. "
        f"Current visualization area is {area_ratio:.2f}x of selected vehicle area. "
        f"Drag is estimated from downstream wake momentum deficit in the pseudo-flow model."
    )

# ==============================================================
# TAB 2: BATTERY ENGINEERING
# ==============================================================
with tab_battery:
    st.subheader("Battery Pack Engineering and Health Prediction")
    btab1, btab2, btab3 = st.tabs(["Pack Configurator", "SOH and Aging", "Thermal Management"])

    with btab1:
        st.markdown("#### Ns x Np Cell Arrangement Architect")
        bc1, bc2, bc3 = st.columns(3)
        v_cell_in  = bc1.number_input("Cell Voltage (V)",    2.0, 4.5, float(p["v_cell"]), 0.05)
        ah_cell_in = bc2.number_input("Cell Capacity (Ah)", 0.5, 300.0, float(p["ah_cell"]), 0.5)
        v_target   = bc3.number_input("Target Pack Voltage (V)", 12.0, 800.0, float(p["v_nom"]), 1.0)
        ah_target  = st.slider("Target Pack Capacity (Ah)", 10, 500, int(p["ah"]))
        chemistry  = st.selectbox("Cell Chemistry",
                                   ["LFP (3.2V, long life)", "NMC (3.7V, high energy)",
                                    "LTO (2.3V, ultra-fast)", "NCA (3.65V, high power)"])

        ns_calc   = max(1, round(v_target / v_cell_in))
        np_calc   = max(1, int(np.ceil(ah_target / ah_cell_in)))
        act_v     = ns_calc * v_cell_in
        act_ah    = np_calc * ah_cell_in
        act_kwh   = (act_v * act_ah) / 1000.0
        chem_cost = {"LFP (3.2V, long life)": 120, "NMC (3.7V, high energy)": 155,
                     "LTO (2.3V, ultra-fast)": 260, "NCA (3.65V, high power)": 165}
        pack_cost_est = act_kwh * chem_cost[chemistry]

        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Configuration", f"{ns_calc}S x {np_calc}P")
        rc2.metric("Total Cells",   f"{ns_calc * np_calc}")
        rc3.metric("Pack Energy",   f"{act_kwh:.2f} kWh")
        rc4.metric("Est. Cost",     f"${pack_cost_est:,.0f}")
        st.success(f"**{ns_calc}S{np_calc}P** | {act_v:.1f} V | {act_ah:.1f} Ah | {act_kwh:.2f} kWh")

        st.markdown("#### Cell Layout Preview")
        disp_ns = min(ns_calc, 24)
        disp_np = min(np_calc, 32)
        fw = max(6, disp_np * 0.45 + 1)
        fh = max(3, disp_ns * 0.35 + 1)
        fig_pk, ax_pk = plt.subplots(figsize=(fw, fh), facecolor=DARK_BG)
        ax_pk.set_facecolor(DARK_BG)
        for row in range(disp_ns):
            for col in range(disp_np):
                rect = plt.Rectangle((col * 1.2, row * 0.65), 1.0, 0.55,
                                      facecolor=C["primary"], edgecolor=GRID_CLR, alpha=0.75)
                ax_pk.add_patch(rect)
        if ns_calc > disp_ns or np_calc > disp_np:
            ax_pk.text(disp_np * 0.6, disp_ns * 0.35,
                        f"Showing {disp_ns}S x {disp_np}P  (full: {ns_calc}S x {np_calc}P)",
                        color="white", ha="center", fontsize=9)
        ax_pk.set_xlim(-0.5, disp_np * 1.2 + 0.5)
        ax_pk.set_ylim(-0.5, disp_ns * 0.65 + 0.5)
        ax_pk.set_xlabel(f"Parallel (Np = {np_calc})", color=TICK_CLR)
        ax_pk.set_ylabel(f"Series (Ns = {ns_calc})",   color=TICK_CLR)
        ax_pk.tick_params(colors=TICK_CLR)
        for sp in ax_pk.spines.values(): sp.set_color(GRID_CLR)
        st.pyplot(fig_pk);  plt.close(fig_pk)

    with btab2:
        st.markdown("#### Battery Degradation and Cycle Life Prediction")
        soh_c1, soh_c2 = st.columns([2, 1])
        with soh_c2:
            max_cyc   = st.number_input("Simulation Cycles", 500, 6000, 2000, 100)
            op_temp   = st.slider("Operating Temperature (C)", -10, 55, 25)
            dod_input = st.slider("Depth of Discharge (%)", 20, 100, 80) / 100.0

        c_arr, soh_arr, r_arr, eol = battery_soh_aging(int(max_cyc), op_temp, dod_input)

        with soh_c1:
            fig_soh, ax_soh = new_fig((8, 4))
            ax_soh.plot(c_arr, soh_arr, color=C["success"], lw=2.5, label="Battery SOH (%)")
            ax_soh.axhline(80, color=C["danger"], ls="--", lw=1.5, label="End-of-Life (80%)")
            ax_soh.fill_between(c_arr, soh_arr, 80, where=(soh_arr >= 80),
                                 alpha=0.12, color=C["success"])
            if eol < max_cyc:
                ax_soh.axvline(eol, color=C["warning"], ls=":", lw=2,
                                label=f"EOL @ {eol:,} cycles")
            ax_soh.set_xlabel("Cycle Count"); ax_soh.set_ylabel("SOH (%)")
            ax_soh.set_title("Capacity Fade Prediction")
            ax_soh.set_ylim(45, 106)
            ax_soh.legend(facecolor=PANEL_BG, labelcolor="white", fontsize=9)
            st.pyplot(fig_soh);  plt.close(fig_soh)

        ec1, ec2, ec3 = st.columns(3)
        ec1.metric("Predicted Cycle Life",  f"{eol:,} cycles")
        ec2.metric("SOH at Sim End",        f"{soh_arr[-1]:.1f}%")
        ec3.metric("Resistance at Sim End", f"{r_arr[-1]:.2f} x R0")

        fig_r, ax_r = new_fig((8, 3))
        ax_r.plot(c_arr, r_arr, color=C["warning"], lw=2)
        ax_r.set_xlabel("Cycles"); ax_r.set_ylabel("Resistance Multiplier (x R0)")
        ax_r.set_title("Internal Resistance Growth")
        st.pyplot(fig_r);  plt.close(fig_r)

    with btab3:
        st.markdown("#### Battery Thermal Model (Lumped Capacitance)")
        th_c1, th_c2 = st.columns(2)
        t_amb_th = th_c1.slider("Ambient Temperature (C)", -20, 45, 25)
        c_rate   = th_c2.slider("Discharge C-Rate", 0.1, 3.0, 1.0, 0.1)
        t_sim_th = np.arange(0, 3600, 1)
        I_prof   = np.ones(len(t_sim_th)) * c_rate * p["ah"]
        temps_th = battery_thermal_model(p, I_prof, t_ambient=t_amb_th)

        fig_th, ax_th = new_fig()
        ax_th.plot(t_sim_th / 60, temps_th, color=C["danger"], lw=2)
        ax_th.fill_between(t_sim_th / 60, temps_th, t_amb_th, alpha=0.15, color=C["danger"])
        ax_th.axhline(45, color=C["warning"], ls="--", lw=1.5, label="Warning (45 C)")
        ax_th.axhline(60, color=C["danger"],  ls=":",  lw=1.5, label="Critical (60 C)")
        ax_th.set_xlabel("Time (min)"); ax_th.set_ylabel("Temperature (C)")
        ax_th.set_title(f"Thermal Profile - {c_rate:.1f}C Discharge")
        ax_th.legend(facecolor=PANEL_BG, labelcolor="white", fontsize=9)
        st.pyplot(fig_th);  plt.close(fig_th)

        peak_t = temps_th.max()
        if peak_t > 60:
            st.error(f"CRITICAL: {peak_t:.1f} C - reduce C-rate or add active cooling.")
        elif peak_t > 45:
            st.warning(f"WARNING: {peak_t:.1f} C - consider thermal management.")
        else:
            st.success(f"Safe thermal operation. Peak: {peak_t:.1f} C")

# ==============================================================
# TAB 3: PERFORMANCE AND RANGE
# ==============================================================
with tab_performance:
    st.subheader("Acceleration, Hill Climbing and Range vs Speed")
    ptab1, ptab2 = st.tabs(["Acceleration and Grade", "Range vs Speed"])

    with ptab1:
        pa_c1, pa_c2 = st.columns(2)
        with pa_c1:
            st.markdown("#### 0-80 km/h Acceleration Simulation")
            peak_kw_in = st.slider("Motor Peak Power (kW)", 1.0, 150.0,
                                    float(max(float(p_motor.max()), 5.0)), 0.5)
            t_acc, v_acc = simulate_acceleration(p, peak_kw_in, target_kmh=80)
            accel_time   = t_acc[-1]

            fig_acc, ax_acc = new_fig()
            ax_acc.plot(t_acc, v_acc, color=C["primary"], lw=2.5)
            ax_acc.fill_between(t_acc, 0, v_acc, alpha=0.15, color=C["primary"])
            ax_acc.set_xlabel("Time (s)"); ax_acc.set_ylabel("Speed (km/h)")
            ax_acc.set_title("Acceleration Profile (0-80 km/h)")
            st.pyplot(fig_acc);  plt.close(fig_acc)
            st.metric("0-80 km/h Time", f"{accel_time:.1f} s")

        with pa_c2:
            st.markdown("#### Grade Climbing Envelope")
            grades_arr  = np.linspace(0, 30, 50)
            max_spd_arr = []
            for gr in grades_arr:
                for v_t in np.linspace(1, 160, 200):
                    pwr_req, _ = motor_power_kw(v_t, p, gr)
                    if pwr_req >= peak_kw_in:
                        max_spd_arr.append(v_t)
                        break
                else:
                    max_spd_arr.append(160)

            fig_gr, ax_gr = new_fig()
            ax_gr.plot(grades_arr, max_spd_arr, color=C["warning"], lw=2.5)
            ax_gr.fill_between(grades_arr, 0, max_spd_arr, alpha=0.15, color=C["warning"])
            ax_gr.set_xlabel("Grade (%)"); ax_gr.set_ylabel("Max Sustained Speed (km/h)")
            ax_gr.set_title("Grade Performance Envelope")
            st.pyplot(fig_gr);  plt.close(fig_gr)

    with ptab2:
        speeds_rng = np.linspace(5, top_speed, 80)
        ranges_arr, wh_km_arr = [], []
        for spd in speeds_rng:
            f_t, fa, fr, fg = tractive_forces(spd, p)
            v_ms = spd / 3.6
            p_w  = f_t * v_ms / p["eta_motor"]
            wh_k = (p_w / max(v_ms, 0.001)) * (1000.0 / 3600.0)
            ranges_arr.append(pack_kwh * 1000.0 / wh_k if wh_k > 0 else 0)
            wh_km_arr.append(wh_k)

        fig_rv, (ax_rv, ax_wh) = plt.subplots(2, 1, figsize=(8, 6), facecolor=DARK_BG)
        for ax in (ax_rv, ax_wh): _apply_dark(ax)
        ax_rv.plot(speeds_rng, ranges_arr, color=C["success"], lw=2.5)
        ax_rv.fill_between(speeds_rng, 0, ranges_arr, alpha=0.12, color=C["success"])
        ax_rv.set_ylabel("Range (km)"); ax_rv.set_title("Range and Consumption vs Speed")
        ax_wh.plot(speeds_rng, wh_km_arr, color=C["secondary"], lw=2.5)
        ax_wh.fill_between(speeds_rng, 0, wh_km_arr, alpha=0.12, color=C["secondary"])
        ax_wh.set_xlabel("Speed (km/h)"); ax_wh.set_ylabel("Consumption (Wh/km)")
        plt.tight_layout()
        st.pyplot(fig_rv);  plt.close(fig_rv)

        opt_idx = int(np.argmax(ranges_arr))
        st.metric("Optimal Efficiency Speed",
                  f"{speeds_rng[opt_idx]:.0f} km/h",
                  delta=f"Max range: {ranges_arr[opt_idx]:.0f} km")

# ==============================================================
# TAB 4: DRIVE CYCLE ANALYSIS
# ==============================================================
with tab_drivecycle:
    st.subheader("Drive Cycle Simulation and Energy Budget")

    dc_c1, dc_c2 = st.columns([1, 2])
    with dc_c1:
        cycle_choice = st.selectbox("Drive Cycle",
                                     ["Urban Stop-Go", "Highway Cruise",
                                      "Mixed Duty", "WLTP-style", "Custom"])
        cycle_dur = st.slider("Duration (s)", 300, 3600, 1200)

        if cycle_choice == "Custom":
            raw_spd = st.text_input("Speed profile (comma-separated km/h)",
                                     "0,20,35,30,45,50,35,15,0,10,30,50,60,45,20,0")
            try:
                spd_vals = [float(x.strip()) for x in raw_spd.split(",")]
                t_cycle  = np.arange(cycle_dur)
                t_interp = np.linspace(0, cycle_dur - 1, len(spd_vals))
                v_cycle  = np.interp(t_cycle, t_interp, spd_vals)
            except Exception:
                st.error("Invalid profile - using Urban Stop-Go instead.")
                t_cycle, v_cycle = generate_drive_cycle("Urban Stop-Go", cycle_dur)
        else:
            t_cycle, v_cycle = generate_drive_cycle(cycle_choice, cycle_dur)

    energy_used, dist_km, c_energy, c_dist = simulate_drive_cycle(v_cycle, p)

    with dc_c2:
        fig_dc, axes_dc = plt.subplots(3, 1, figsize=(9, 7), facecolor=DARK_BG)
        t_min = t_cycle / 60
        for ax in axes_dc: _apply_dark(ax)

        axes_dc[0].fill_between(t_min, v_cycle, alpha=0.5, color=C["primary"])
        axes_dc[0].plot(t_min, v_cycle, color=C["primary"], lw=1.5)
        axes_dc[0].set_ylabel("Speed (km/h)")
        axes_dc[0].set_title("Drive Cycle Analysis")

        axes_dc[1].fill_between(t_min[:len(c_energy)], c_energy, alpha=0.5, color=C["secondary"])
        axes_dc[1].set_ylabel("Cumulative Energy (Wh)")

        soc_profile = np.clip(100 - (c_energy / (pack_kwh * 1000)) * 100, 0, 100)
        axes_dc[2].plot(t_min[:len(soc_profile)], soc_profile, color=C["success"], lw=2)
        axes_dc[2].axhline(20, color=C["danger"], ls="--", lw=1, label="Low SOC threshold")
        axes_dc[2].set_ylabel("SOC (%)"); axes_dc[2].set_xlabel("Time (min)")
        axes_dc[2].legend(facecolor=PANEL_BG, labelcolor="white", fontsize=8)

        plt.tight_layout()
        st.pyplot(fig_dc);  plt.close(fig_dc)

    dc_m1, dc_m2, dc_m3, dc_m4 = st.columns(4)
    dc_m1.metric("Distance Covered",     f"{dist_km:.2f} km")
    dc_m2.metric("Energy Consumed",      f"{energy_used:.1f} Wh")
    eff_wh = energy_used / dist_km if dist_km > 0 else 0
    dc_m3.metric("Avg Consumption",      f"{eff_wh:.1f} Wh/km" if eff_wh > 0 else "N/A")
    proj_range = pack_kwh * 1000 / eff_wh if eff_wh > 0 else 0
    dc_m4.metric("Projected Full Range", f"{proj_range:.1f} km" if proj_range > 0 else "N/A")

# ==============================================================
# TAB 5: ECONOMICS AND ENVIRONMENT
# ==============================================================
with tab_economics:
    st.subheader("Total Cost of Ownership and Carbon Impact")
    etab1, etab2 = st.tabs(["TCO Calculator", "Carbon and Credits"])

    with etab1:
        st.markdown("#### EV vs ICE - Lifetime Cost Analysis")
        e1, e2, e3 = st.columns(3)
        ev_price  = e1.number_input("EV Purchase Price ($)",    500, 200000, 8000)
        ice_price = e2.number_input("ICE Equivalent Price ($)", 500, 100000, 5000)
        years_n   = e3.number_input("Analysis Period (years)",   1,  20,     10)

        e4, e5, e6 = st.columns(3)
        elec_price = e4.number_input("Electricity ($/kWh)", 0.04, 0.60, 0.12, 0.01)
        fuel_price = e5.number_input("Fuel ($/L)",          0.40, 3.00, 1.00, 0.05)
        ann_km     = e6.number_input("Annual Mileage (km)", 1000, 100000, 12000)

        e7, e8 = st.columns(2)
        bat_cpkwh = e7.number_input("Battery Cost ($/kWh)", 50, 600, 150)
        ice_l100  = e8.number_input("ICE Fuel Use (L/100km)", 2.0, 15.0, 4.0, 0.5)

        tco = tco_analysis(p, {
            "years": int(years_n), "annual_km": ann_km, "wh_km": wh_km,
            "elec_price": elec_price, "fuel_price": fuel_price,
            "ev_price": ev_price, "ice_price": ice_price,
            "bat_cost_kwh": bat_cpkwh, "l_100km": ice_l100,
        })

        t1, t2, t3, t4 = st.columns(4)
        t1.metric("EV Lifetime Cost",  f"${tco['ev_total']:,.0f}")
        t2.metric("ICE Lifetime Cost", f"${tco['ice_total']:,.0f}")
        t3.metric("Lifetime Savings",  f"${tco['savings']:,.0f}", delta="vs ICE")
        t4.metric("Breakeven Year",
                  f"Year {tco['breakeven']}" if tco["breakeven"] else "Beyond period")

        fig_tco, ax_tco = new_fig((9, 4))
        ax_tco.plot(tco["yr"], tco["ev_cum"],  color=C["primary"],   lw=2.5,
                    marker="o", markevery=2, markersize=4, label="EV")
        ax_tco.plot(tco["yr"], tco["ice_cum"], color=C["secondary"], lw=2.5,
                    marker="s", markevery=2, markersize=4, label="ICE")
        if tco["breakeven"]:
            ax_tco.axvline(tco["breakeven"], color=C["success"], ls="--", lw=1.5,
                           label=f"Breakeven: Year {tco['breakeven']}")
        ax_tco.set_xlabel("Year"); ax_tco.set_ylabel("Cumulative Cost ($)")
        ax_tco.set_title("Total Cost of Ownership - EV vs ICE")
        ax_tco.legend(facecolor=PANEL_BG, labelcolor="white", fontsize=9)
        st.pyplot(fig_tco);  plt.close(fig_tco)

    with etab2:
        st.markdown("#### Carbon Footprint Comparison")
        env1, env2 = st.columns(2)
        ann_km_env   = env1.number_input("Annual Mileage (km)",     1000, 100000, 12000, key="env_km")
        grid_factor  = env2.slider("Grid Carbon Intensity (kgCO2/kWh)", 0.05, 1.0, 0.45, 0.05)
        ice_l100_env = env1.number_input("ICE Fuel Use (L/100km)", 2.0, 15.0, 4.0, 0.5, key="env_l")
        fuel_co2_map = {
            "Petrol (2.31 kgCO2/L)": 2.31,
            "Diesel (2.68 kgCO2/L)": 2.68,
            "CNG (1.87 kgCO2/kg)":   1.87,
        }
        fuel_type_env = env2.selectbox("ICE Fuel Type", list(fuel_co2_map.keys()))
        fuel_co2_fac  = fuel_co2_map[fuel_type_env]

        ev_co2_yr  = (wh_km / 1000) * grid_factor * ann_km_env
        ice_co2_yr = (ice_l100_env / 100) * fuel_co2_fac * ann_km_env
        saved_yr   = ice_co2_yr - ev_co2_yr
        saved_life = saved_yr * int(years_n)

        ce1, ce2, ce3, ce4 = st.columns(4)
        ce1.metric("EV Annual CO2",   f"{ev_co2_yr:.0f} kg")
        ce2.metric("ICE Annual CO2",  f"{ice_co2_yr:.0f} kg")
        ce3.metric("Annual Saving",   f"{saved_yr:.0f} kg CO2")
        ce4.metric("Lifetime Saving", f"{saved_life:.0f} kg CO2")

        st.info(f"{saved_yr:.0f} kg CO2 saved/year is equivalent to planting {int(saved_yr / 21)} trees per year.")

        fig_env, ax_env = new_fig((8, 4))
        categories = ["EV (grid)", "EV (100% renewable)", "ICE Petrol", "ICE Diesel"]
        co2_vals   = [
            ev_co2_yr, 0,
            (ice_l100_env / 100) * 2.31 * ann_km_env,
            (ice_l100_env / 100) * 2.68 * ann_km_env,
        ]
        bar_colors = [C["primary"], C["success"], C["secondary"], C["danger"]]
        bars = ax_env.bar(categories, co2_vals, color=bar_colors, edgecolor=GRID_CLR, width=0.5)
        for bar, val in zip(bars, co2_vals):
            ax_env.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 25,
                        f"{val:.0f}", ha="center", va="bottom", color="white", fontsize=9)
        ax_env.set_ylabel("Annual CO2 Emissions (kg)")
        ax_env.set_title("Carbon Footprint - Annual Comparison")
        st.pyplot(fig_env);  plt.close(fig_env)

# ==============================================================
# TAB 6: EV ARCHITECTURES
# ==============================================================
with tab_architecture:
    st.subheader("BEV, HEV, PHEV and EREV Architecture Studio")
    arch_tab1, arch_tab2 = st.tabs(["BEV Types", "Hybrid Topologies"])

    with arch_tab1:
        st.markdown("#### BEV (Battery Electric Vehicle) Types")
        bev_options = {
            "Urban 2W/3W BEV": {
                "Battery": "1.5 to 8 kWh",
                "Motor": "2 to 12 kW",
                "Use": "Last-mile, delivery, low operating cost",
                "Charging": "Slow AC + battery swap in some fleets",
            },
            "Passenger Car BEV": {
                "Battery": "25 to 100 kWh",
                "Motor": "60 to 300+ kW",
                "Use": "City and highway personal mobility",
                "Charging": "AC home + DC fast charging",
            },
            "Commercial Van/Bus BEV": {
                "Battery": "80 to 450 kWh",
                "Motor": "120 to 450 kW",
                "Use": "Fleet routes, public transport, logistics",
                "Charging": "Depot charging, pantograph, opportunity charging",
            },
            "Performance AWD BEV": {
                "Battery": "70 to 120 kWh",
                "Motor": "Dual/triple motor, 300 to 800+ kW",
                "Use": "High acceleration and dynamic handling",
                "Charging": "High-power DC + preconditioning",
            },
        }
        bev_pick = st.selectbox("Select BEV Type", list(bev_options.keys()))
        bev_info = bev_options[bev_pick]

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Typical Battery", bev_info["Battery"])
        b2.metric("Motor Range", bev_info["Motor"])
        b3.metric("Primary Use", bev_info["Use"])
        b4.metric("Charging", bev_info["Charging"])

        st.markdown("#### Typical BEV Drivetrain Layouts")
        layout_df = pd.DataFrame({
            "Layout": ["Single Motor FWD", "Single Motor RWD", "Dual Motor AWD", "In-wheel Motors"],
            "Efficiency": ["High", "High", "Medium-High", "Medium"],
            "Complexity": ["Low", "Low", "Medium", "High"],
            "Key Strength": [
                "Low cost, compact packaging",
                "Balanced handling and towing",
                "Traction and performance",
                "Independent torque control"
            ],
        })
        st.dataframe(layout_df, use_container_width=True, hide_index=True)

        st.info(
            "BEV means propulsion is always electric. There is no engine-driven mechanical path to wheels."
        )

    with arch_tab2:
        st.markdown("#### HEV/PHEV/EREV Transmission and Operating Modes")
        h1, h2 = st.columns(2)
        vehicle_class = h1.selectbox("Electrified Vehicle Type", ["HEV", "PHEV", "EREV"])
        topology_options = {
            "HEV": ["Series", "Parallel", "Series-Parallel"],
            "PHEV": ["Series", "Parallel", "Series-Parallel"],
            "EREV": ["Series"],
        }
        topology = h2.selectbox("Transmission Topology", topology_options[vehicle_class])

        fig_arch = draw_architecture_diagram(vehicle_class, topology)
        st.pyplot(fig_arch)
        plt.close(fig_arch)

        explain_map = {
            ("HEV", "Series"): "Engine drives generator only. Wheels are driven by motor. Best in stop-go cycles.",
            ("HEV", "Parallel"): "Engine and motor can both drive wheels through transmission. Efficient at cruise.",
            ("HEV", "Series-Parallel"): "Power-split unit blends series and parallel paths for best overall efficiency.",
            ("PHEV", "Series"): "Like series HEV but with larger battery and grid charging for EV-first operation.",
            ("PHEV", "Parallel"): "Motor + engine both provide wheel torque; charge port enables long EV-only range.",
            ("PHEV", "Series-Parallel"): "Most flexible PHEV architecture, balancing EV mode and highway efficiency.",
            ("EREV", "Series"): "Wheels are always motor-driven. Engine acts only as range extender generator.",
        }
        st.success(explain_map[(vehicle_class, topology)])

        mode_table = {
            ("HEV", "Series"): [
                ["Low speed", "Battery -> Motor", "Electric launch"],
                ["Cruise", "ICE -> Generator -> Motor", "Engine held near efficient zone"],
                ["Braking", "Wheels -> Motor -> Battery", "Regen recovery"],
            ],
            ("HEV", "Parallel"): [
                ["Launch", "Battery -> Motor (+ optional ICE)", "Torque assist"],
                ["Cruise", "ICE -> Transmission -> Wheels", "Highway efficient"],
                ["Overtake", "ICE + Motor -> Wheels", "Combined torque"],
            ],
            ("HEV", "Series-Parallel"): [
                ["City", "Battery + motor prioritized", "Lower fuel use"],
                ["Mixed", "Split power path", "Controller optimizes engine load"],
                ["Braking", "Regen + engine decoupling", "Energy recovery"],
            ],
            ("PHEV", "Series"): [
                ["EV mode", "Battery -> Motor", "Daily commute without fuel"],
                ["Charge-sustain", "ICE -> Generator -> Motor", "Range extension after EV depletion"],
                ["Braking", "Regen to battery", "Higher recovered energy due to larger pack"],
            ],
            ("PHEV", "Parallel"): [
                ["EV mode", "Battery -> Motor -> Wheels", "Pure electric until SOC target"],
                ["Hybrid mode", "ICE + Motor -> Wheels", "Performance and efficiency blend"],
                ["Charge mode", "ICE charges battery while driving", "Optional SOC management"],
            ],
            ("PHEV", "Series-Parallel"): [
                ["Urban", "Mostly electric", "Engine off when possible"],
                ["Highway", "Parallel dominant", "Reduced conversion losses"],
                ["Transient", "Series assist + battery buffering", "Flexible control"],
            ],
            ("EREV", "Series"): [
                ["EV-first", "Battery -> Motor", "Primary operating mode"],
                ["Range extension", "ICE -> Generator -> Motor", "No engine-to-wheel mechanical link"],
                ["Braking", "Regen to battery", "Maintains buffer SOC"],
            ],
        }
        mode_df = pd.DataFrame(
            mode_table[(vehicle_class, topology)],
            columns=["Driving Condition", "Energy Path", "What Happens"],
        )
        st.dataframe(mode_df, use_container_width=True, hide_index=True)

        compare_df = pd.DataFrame({
            "Type": ["HEV", "PHEV", "EREV"],
            "Grid Charging": ["No", "Yes", "Yes"],
            "Typical EV-Only Range": ["0-5 km", "30-120 km", "60-200 km"],
            "Engine Role": ["Primary + assist", "Primary + electric-first", "Generator only"],
        })
        st.markdown("#### Quick Comparison")
        st.dataframe(compare_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### Degree of Hybridization Calculator")
        hc1, hc2, hc3 = st.columns(3)
        motor_in = hc1.number_input("Motor Power (kW)", 0.0, 500.0, 35.0, 1.0)
        battery_in = hc2.number_input("Battery Energy (kWh)", 0.0, 200.0, 1.8, 0.1)
        ice_in = hc3.number_input("ICE Power (kW)", 0.0, 500.0, 75.0, 1.0)

        hyb = classify_hybridization(motor_in, battery_in, ice_in)

        hm1, hm2, hm3 = st.columns(3)
        hm1.metric("Hybrid Type", hyb["type"])
        hm2.metric("Degree of Hybridization", f"{hyb['doh_pct']:.1f}%")
        hm3.metric("Total Propulsion Power", f"{hyb['total_prop']:.1f} kW")
        st.progress(min(max(hyb["doh"], 0.0), 1.0))
        st.info(hyb["note"])

        fig_h, ax_h = new_fig((8, 3.8))
        shares = [ice_in, motor_in]
        labels = ["ICE", "Motor"]
        colors = [C["secondary"], C["primary"]]
        ax_h.barh(["Power Share"], [shares[0]], color=colors[0], edgecolor=GRID_CLR, label=labels[0])
        ax_h.barh(["Power Share"], [shares[1]], left=[shares[0]], color=colors[1], edgecolor=GRID_CLR,
                  label=labels[1])
        ax_h.set_xlabel("Installed Power (kW)")
        ax_h.set_title("Installed Propulsion Split (ICE vs Motor)")
        ax_h.legend(facecolor=PANEL_BG, labelcolor="white", fontsize=8, loc="lower right")
        st.pyplot(fig_h)
        plt.close(fig_h)

        rules_df = pd.DataFrame({
            "Category": ["Mild Hybrid", "Full Hybrid", "Plug-in Hybrid"],
            "Typical DoH": ["8-28%", "28-55%", ">= 20% (with large battery)"],
            "Battery Guideline": ["0.3-2 kWh", "1-8 kWh", ">= 8 kWh"],
            "EV-Only Capability": ["Very limited", "Low to moderate", "High (commute-capable)"],
        })
        st.caption("Classification is rule-based and intended for early architecture screening.")
        st.dataframe(rules_df, use_container_width=True, hide_index=True)

# ==============================================================
# TAB 7: DATA ANALYTICS
# ==============================================================
with tab_analytics:
    st.subheader("Synthetic Telemetry and Performance Dashboard")
    at1, at2 = st.tabs(["Telemetry Dashboard", "Export and Report"])

    with at1:
        seed_v = st.slider("Trip Seed (vary for different simulated trips)", 0, 100, 42)
        np.random.seed(seed_v)

        t_tel    = np.arange(0, 3601, 5)
        v_tel    = np.clip(
            40 + 25 * np.sin(2 * np.pi * t_tel / 600) + 8 * np.random.randn(len(t_tel)),
            0, 100)
        v_ms_tel   = v_tel / 3.6
        f_aero_tel = 0.5 * RHO_AIR * p["cd"] * p["area"] * v_ms_tel ** 2
        f_roll_tel = p["m"] * G * p["cr"]
        p_mt_tel   = ((f_aero_tel + f_roll_tel) * v_ms_tel) / (p["eta_motor"] * 1000)
        accel_tel  = np.concatenate(([0.0], np.diff(v_ms_tel) / 5.0))
        soc_tel    = np.clip(
            100 - np.cumsum(p_mt_tel) * 5 / 3600 / pack_kwh, 0, 100)
        temp_delta = np.cumsum(np.abs(p_mt_tel)) * 5 / (p["battery_thermal_mass"] * 900) * 0.3
        temp_tel   = 25 + temp_delta

        df_tel = pd.DataFrame({
            "Time (s)":            t_tel,
            "Speed (km/h)":        v_tel,
            "Motor Power (kW)":    p_mt_tel,
            "SOC (%)":             soc_tel,
            "Temperature (C)":     temp_tel,
            "Acceleration (m/s2)": accel_tel,
        })

        fig_dash, axes_d = plt.subplots(3, 2, figsize=(12, 9), facecolor=DARK_BG)
        plot_cfg = [
            ("Speed (km/h)",        C["primary"]),
            ("Motor Power (kW)",    C["secondary"]),
            ("SOC (%)",             C["success"]),
            ("Temperature (C)",     C["danger"]),
            ("Acceleration (m/s2)", C["warning"]),
        ]
        t_min_tel = df_tel["Time (s)"] / 60
        for i, (col_name, clr) in enumerate(plot_cfg):
            ax = axes_d.flatten()[i]
            _apply_dark(ax)
            ax.plot(t_min_tel, df_tel[col_name], color=clr, lw=1.2)
            ax.fill_between(t_min_tel, df_tel[col_name], alpha=0.10, color=clr)
            ax.set_ylabel(col_name, color=TICK_CLR, fontsize=8)
            ax.tick_params(labelsize=7)

        ax_hist = axes_d.flatten()[5]
        _apply_dark(ax_hist)
        ax_hist.hist(df_tel["Speed (km/h)"], bins=20,
                     color=C["primary"], edgecolor=GRID_CLR, alpha=0.85)
        ax_hist.set_xlabel("Speed (km/h)", color=TICK_CLR, fontsize=8)
        ax_hist.set_ylabel("Frequency",    color=TICK_CLR, fontsize=8)
        ax_hist.tick_params(labelsize=7)

        plt.suptitle("Telemetry Dashboard", color="white", fontsize=13)
        plt.tight_layout()
        st.pyplot(fig_dash);  plt.close(fig_dash)

        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric("Avg Speed",        f"{df_tel['Speed (km/h)'].mean():.1f} km/h")
        sm2.metric("Peak Motor Power", f"{df_tel['Motor Power (kW)'].max():.1f} kW")
        sm3.metric("Energy Used",
                   f"{(df_tel['Motor Power (kW)'].sum() * 5 / 3600):.2f} kWh")
        sm4.metric("Final SOC",        f"{soc_tel[-1]:.1f}%")

    with at2:
        st.markdown("#### Vehicle Summary and Data Export")

        summary_df = pd.DataFrame({
            "Parameter": [
                "Vehicle Preset", "Mass", "Pack Voltage", "Pack Capacity", "Pack Energy",
                "Drag Coefficient (Cd)", "Rolling Resistance (Cr)", "Motor Efficiency",
                "Regen Efficiency", "Estimated Range", "Motor Power @80 km/h",
            ],
            "Value": [
                preset_choice, f"{p['m']} kg", f"{p['v_nom']} V", f"{p['ah']} Ah",
                f"{pack_kwh:.2f} kWh", str(p["cd"]), str(p["cr"]),
                f"{p['eta_motor']*100:.0f}%", f"{p['eta_regen']*100:.0f}%",
                f"{est_range:.1f} km", f"{p_motor_80:.1f} kW",
            ],
        })
        st.table(summary_df)

        csv_bytes = df_tel.to_csv(index=False).encode()
        st.download_button(
            label="Download Telemetry CSV",
            data=csv_bytes,
            file_name=f"ev_telemetry_{preset_choice.replace(' ', '_')}.csv",
            mime="text/csv",
        )

        txt_report = (
            f"EV Design Studio Pro - Vehicle Report\n"
            f"======================================\n"
            f"Preset       : {preset_choice}\n"
            f"Mass         : {p['m']} kg\n"
            f"Pack         : {p['v_nom']} V x {p['ah']} Ah = {pack_kwh:.2f} kWh\n"
            f"Cd / Cr      : {p['cd']} / {p['cr']}\n"
            f"Motor eff    : {p['eta_motor']*100:.0f}%\n"
            f"Regen eff    : {p['eta_regen']*100:.0f}%\n"
            f"Range (est.) : {est_range:.1f} km  [{drive_style}]\n"
            f"Power@80km/h : {p_motor_80:.1f} kW\n"
            f"Tractive F   : {tractive_forces(80, p)[0]:.0f} N @ 80 km/h\n"
        )
        st.download_button(
            label="Download Summary TXT",
            data=txt_report.encode(),
            file_name=f"ev_report_{preset_choice.replace(' ', '_')}.txt",
            mime="text/plain",
        )
