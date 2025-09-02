import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

# ===============================
# Funzioni di simulazione batteria
# ===============================
def simula_batteria_carico(
    E_step_kWh,
    E_target_step_kWh,
    E_nom_kWh,
    E_step_limit_kWh,
    eta_charge=0.95,
    eta_discharge=0.95,
    DoD=0.9,
    soc_init_frac=0.5,
):
    E_usable = E_nom_kWh * DoD
    soc = soc_init_frac * E_usable
    E_dispatched_total = 0.0
    E_charged_total = 0.0
    sum_excess = 0.0
    soc_series = []

    for E in E_step_kWh:
        excess = max(0, E - E_target_step_kWh)
        deficit = max(0, E_target_step_kWh - E)

        if excess > 0 and soc > 0:
            E_deliverable = min(excess, E_step_limit_kWh, soc * eta_discharge)
            soc -= E_deliverable / eta_discharge
            E_dispatched_total += E_deliverable

        if deficit > 0 and soc < E_usable:
            E_storable = min(deficit, E_step_limit_kWh) * eta_charge
            spazio = E_usable - soc
            E_stored = min(E_storable, spazio)
            soc += E_stored
            E_charged_total += E_stored

        sum_excess += excess
        soc_series.append(soc)

    residual_excess = max(0, sum_excess - E_dispatched_total)

    return {
        "E_dispatched_total": E_dispatched_total,
        "E_charged_total": E_charged_total,
        "sum_excess": sum_excess,
        "residual_excess": residual_excess,
        "soc_series": soc_series,
    }

def trova_capacity_per_target_carico(
    E_step_kWh,
    E_target_step_kWh,
    E_step_limit_kWh,
    eta_charge=0.95,
    eta_discharge=0.95,
    DoD=0.9,
    tol_kwh=1e-3,
    soc_init_frac=0.5,
    max_iter=40,
):
    sum_excess = sum(max(0, e - E_target_step_kWh) for e in E_step_kWh)
    if sum_excess == 0:
        return {"E_nom_kWh": 0.0, "E_usable_kWh": 0.0, "sim": None}

    upper = max(1.0, sum_excess / DoD) * 1.2
    lo, hi = 0.0, upper
    best = None

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        sim = simula_batteria_carico(
            E_step_kWh,
            E_target_step_kWh,
            mid,
            E_step_limit_kWh,
            eta_charge,
            eta_discharge,
            DoD,
            soc_init_frac,
        )
        if sim["residual_excess"] <= tol_kwh:
            best = (mid, sim)
            hi = mid
        else:
            lo = mid

    if best is None:
        E_nom_kWh = hi
        sim = simula_batteria_carico(
            E_step_kWh,
            E_target_step_kWh,
            E_nom_kWh,
            E_step_limit_kWh,
            eta_charge,
            eta_discharge,
            DoD,
            soc_init_frac,
        )
        best = (E_nom_kWh, sim)

    E_nom_kWh, sim = best
    return {"E_nom_kWh": E_nom_kWh, "E_usable_kWh": E_nom_kWh * DoD, "sim": sim}

# ===============================
# Funzione obiettivo per ottimizzazione soglia
# ===============================
def rapporto_batteria_soglia(soglia, E_step, E_step_limit, eta_charge, eta_discharge, DoD, tol_kwh, soc_init_frac):
    res = trova_capacity_per_target_carico(
        E_step,
        soglia,
        E_step_limit,
        eta_charge,
        eta_discharge,
        DoD,
        tol_kwh,
        soc_init_frac
    )
    E_nom = res["E_nom_kWh"]
    return E_nom / soglia if soglia > 0 else np.inf

# ===============================
# Ottimizzazione soglia per ogni mese
# ===============================
def ottimizza_soglia_per_mese(
    df_carico,
    eta_charge=0.95,
    eta_discharge=0.95,
    DoD=0.9,
    tol_kwh=1e-3,
    soc_init_frac=0.5,
    soglia_min=50,
    soglia_max=300
):
    df = df_carico.copy()
    df["Time"] = pd.to_datetime(df["Time"])
    df["Mese"] = df["Time"].dt.to_period("M").astype(str)

    risultati = []

    for mese, g in df.groupby("Mese"):
        E_step = g["Carico_energia"].values
        E_step_limit = np.max(E_step)

        res_opt = minimize_scalar(
            rapporto_batteria_soglia,
            bounds=(soglia_min, soglia_max),
            args=(E_step, E_step_limit, eta_charge, eta_discharge, DoD, tol_kwh, soc_init_frac),
            method='bounded'
        )

        soglia_ottimale = res_opt.x
        res_final = trova_capacity_per_target_carico(
            E_step,
            soglia_ottimale,
            E_step_limit,
            eta_charge,
            eta_discharge,
            DoD,
            tol_kwh,
            soc_init_frac
        )

        E_nom = res_final["E_nom_kWh"]
        ratio = E_nom / soglia_ottimale if soglia_ottimale > 0 else np.nan

        risultati.append({
            "Mese": mese,
            "soglia_ottimale_kWh": soglia_ottimale,
            "E_nom_kWh": E_nom,
            "E_usable_kWh": res_final["E_usable_kWh"],
            "ratio_capacita_soglia": ratio,
            "residual_excess_kWh": res_final["sim"]["residual_excess"] if res_final["sim"] else 0
        })

    return pd.DataFrame(risultati)

# ===============================
# Esempio di utilizzo
# ===============================
df_carico = pd.read_csv("dataset_full.csv")
pp = max(df_carico['Carico_energia'])

risultati_ottimizzati = ottimizza_soglia_per_mese(
    df_carico,
    eta_charge=0.95,
    eta_discharge=0.95,
    DoD=0.9,
    tol_kwh=1e-3,
    soc_init_frac=0.5,
    soglia_min=50,
    soglia_max=600
)

print(risultati_ottimizzati[[
    "Mese", "soglia_ottimale_kWh", "E_nom_kWh", "ratio_capacita_soglia", "residual_excess_kWh"
]])

# Ordiniamo per mese
risultati_ottimizzati["Mese"] = pd.to_datetime(risultati_ottimizzati["Mese"].astype(str) + "-01")
risultati_ottimizzati.sort_values("Mese", inplace=True)

mesi = risultati_ottimizzati["Mese"].dt.strftime("%Y-%m")

# --- Grafico soglia e capacità ---
fig, ax1 = plt.subplots(figsize=(12, 6))

color1 = 'tab:blue'
ax1.set_xlabel('Mese')
ax1.set_ylabel('Soglia ottimale [kWh]', color=color1)
ax1.plot(mesi, risultati_ottimizzati["soglia_ottimale_kWh"], color=color1, marker='o', label='Soglia ottimale')
ax1.tick_params(axis='y', labelcolor=color1)
plt.xticks(rotation=45)

ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Capacità batteria [kWh]', color=color2)
ax2.plot(mesi, risultati_ottimizzati["E_nom_kWh"], color=color2, marker='x', label='E_nom')
ax2.tick_params(axis='y', labelcolor=color2)

fig.tight_layout()
plt.title("Soglia ottimale e capacità batteria per mese")
plt.show()

# --- Grafico rapporto batteria/soglia ---
plt.figure(figsize=(12, 4))
plt.plot(mesi, risultati_ottimizzati["ratio_capacita_soglia"], marker='s', color='tab:green')
plt.xticks(rotation=45)
plt.ylabel("Rapporto batteria / soglia")
plt.xlabel("Mese")
plt.title("Rapporto batteria/soglia per mese")
plt.grid(True)
plt.tight_layout()
plt.show()
