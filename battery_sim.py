import numpy as np
import pandas as pd


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
    """
    Simula l'uso della batteria su un profilo di carico (energia per step, kWh).

    Parametri
    ---------
    E_step_kWh : array-like
        Serie temporale del carico (kWh per step).
    E_target_step_kWh : float
        Soglia desiderata per ogni step (kWh).
    E_nom_kWh : float
        Capacità nominale della batteria (kWh).
    E_step_limit_kWh : float
        Energia massima che la batteria può caricare/scaricare in un singolo step.
        (potenza max inverter * durata step).
    eta_charge : float
        Efficienza di carica.
    eta_discharge : float
        Efficienza di scarica.
    DoD : float
        Depth of Discharge (es. 0.9 = 90% della capacità nominale utilizzabile).
    soc_init_frac : float
        Stato di carica iniziale (frazione della capacità utilizzabile).

    Ritorna
    -------
    dict con:
        - E_dispatched_total
        - E_charged_total
        - sum_excess
        - residual_excess
        - soc_series
    """
    E_usable = E_nom_kWh * DoD
    soc = soc_init_frac * E_usable

    E_dispatched_total = 0.0
    E_charged_total = 0.0
    sum_excess = 0.0
    soc_series = []

    for E in E_step_kWh:
        excess = max(0, E - E_target_step_kWh)
        deficit = max(0, E_target_step_kWh - E)

        # SCARICA: riduco gli excess
        if excess > 0 and soc > 0:
            E_deliverable = min(
                excess,
                E_step_limit_kWh,
                soc * eta_discharge
            )
            soc -= E_deliverable / eta_discharge
            E_dispatched_total += E_deliverable

        # CARICA: se c'è deficit sotto la soglia
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

'''def simula_batteria_carico(
    E_step_kWh,
    E_target_step_kWh,
    E_nom_kWh,
    E_step_limit_kWh,
    eta_charge=0.95,
    eta_discharge=0.95,
    DoD=0.9,
    soc_init_frac=0.5,
):
    """
    Prima strategia (peak shaving con target), estesa per permettere
    anche la carica da rete quando il carico è sotto soglia.
    """

    E_usable = E_nom_kWh * DoD
    soc = soc_init_frac * E_usable

    E_dispatched_total = 0.0
    E_charged_total = 0.0
    sum_excess = 0.0
    soc_series = []

    for E in E_step_kWh:
        excess = max(0, E - E_target_step_kWh)   # quanto sfora sopra la soglia
        deficit = max(0, E_target_step_kWh - E)  # quanto resta sotto la soglia

        # --- SCARICO (peak shaving sopra target) ---
        if excess > 0 and soc > 0:
            E_deliverable = min(excess, E_step_limit_kWh, soc * eta_discharge)
            soc -= E_deliverable / eta_discharge
            E_dispatched_total += E_deliverable

        # --- CARICA (quando sotto target, dalla rete) ---
        if deficit > 0 and soc < E_usable:
            # prelevo dalla rete solo quanto serve per colmare il deficit
            E_from_grid = min(deficit, E_step_limit_kWh)
            # in batteria entra meno, a causa dell'efficienza
            E_stored = min(E_from_grid * eta_charge, E_usable - soc)
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
    }'''


def simula_batteria_peakshaving_autoconsumo(
    E_load_step_kWh,        # array kWh/step: carico
    E_pv_step_kWh,          # array kWh/step: produzione FV
    E_target_step_kWh,      # soglia max import da rete per step (peak shaving)
    E_nom_kWh,              # capacità nominale batteria
    E_step_limit_kWh,       # energia max per step in carica/scarica (potenza * durata step)
    eta_charge=0.95,
    eta_discharge=0.95,
    DoD=0.9,
    soc_init_frac=0.5,
):
    E_load = np.asarray(E_load_step_kWh, dtype=float)
    E_pv   = np.asarray(E_pv_step_kWh, dtype=float)
    assert E_load.shape == E_pv.shape, "E_load_step_kWh e E_pv_step_kWh devono avere stessa lunghezza."

    E_usable = E_nom_kWh * DoD
    soc = soc_init_frac * E_usable
    n = len(E_load)

    soc_series = np.zeros(n)
    batt_charge_stored = np.zeros(n)
    batt_discharge_out = np.zeros(n)
    grid_import = np.zeros(n)
    grid_export = np.zeros(n)
    pv_self_consumption = np.zeros(n)
    pv_to_battery_input = np.zeros(n)

    sum_excess = 0.0      # totale carico sopra soglia
    E_dispatched_total = 0.0

    for i in range(n):
        load = E_load[i]
        pv = E_pv[i]

        direct = min(load, pv)
        pv_self_consumption[i] = direct
        net = load - pv  # >0: residuo carico; <0: surplus FV

        # peak shaving
        discharge = 0.0
        if net > E_target_step_kWh and soc > 0:
            required = net - E_target_step_kWh
            deliverable = min(required, E_step_limit_kWh, soc * eta_discharge)
            soc -= deliverable / eta_discharge
            discharge = deliverable
            net -= deliverable
            E_dispatched_total += deliverable

        # autoconsumo FV
        charge_stored = 0.0
        pv_to_batt_in = 0.0
        if net < 0 and soc < E_usable:
            surplus = -net
            batt_in = min(surplus, E_step_limit_kWh)
            space = E_usable - soc
            stored = min(batt_in * eta_charge, space)
            used_from_pv = stored / eta_charge if eta_charge > 0 else 0.0
            soc += stored
            charge_stored = stored
            pv_to_batt_in = used_from_pv
            net += used_from_pv

        if net >= 0:
            grid_import[i] = net
        else:
            grid_export[i] = -net

        batt_discharge_out[i] = discharge
        batt_charge_stored[i] = charge_stored
        pv_to_battery_input[i] = pv_to_batt_in
        soc_series[i] = soc

        sum_excess += max(0, load - E_target_step_kWh)

    residual_excess = max(0, sum_excess - E_dispatched_total)

    results = {
        "soc_series": soc_series,
        "batt_charge_stored_kWh": batt_charge_stored,
        "batt_discharge_out_kWh": batt_discharge_out,
        "grid_import_kWh": grid_import,
        "grid_export_kWh": grid_export,
        "pv_self_consumption_kWh": pv_self_consumption,
        "pv_to_battery_input_kWh": pv_to_battery_input,
        # aggregati
        "tot_grid_import_kWh": grid_import.sum(),
        "tot_grid_export_kWh": grid_export.sum(),
        "tot_pv_self_consumption_kWh": pv_self_consumption.sum(),
        "tot_pv_to_battery_input_kWh": pv_to_battery_input.sum(),
        "tot_batt_charge_stored_kWh": batt_charge_stored.sum(),
        "tot_batt_discharge_out_kWh": batt_discharge_out.sum(),
        # nuovo campo residual_excess
        "residual_excess": residual_excess,
        "sum_excess": sum_excess,
        "E_dispatched_total": E_dispatched_total
    }
    return results



def simula_batteria_peakshaving_gridcharge(
    E_load_step_kWh,
    E_pv_step_kWh,
    E_target_step_kWh,
    E_nom_kWh,
    E_step_limit_kWh,
    eta_charge=0.95,
    eta_discharge=0.95,
    DoD=0.9,
    soc_init_frac=0.5,
):
    """
    Simula gestione batteria con:
    - FV usato solo per autoconsumo diretto (no carica batteria).
    - Batteria caricata solo dalla rete.
    - Batteria scarica per peak shaving se il carico residuo supera la soglia.
    """

    E_usable = E_nom_kWh * DoD
    soc = soc_init_frac * E_usable

    soc_series = []
    batt_discharge_out = []
    batt_charge_stored = []
    grid_import = []
    grid_export = []
    pv_selfcons = []

    for E_load, E_pv in zip(E_load_step_kWh, E_pv_step_kWh):
        # 1. Autoconsumo FV diretto
        E_pv_used = min(E_load, E_pv)
        E_residual_load = E_load - E_pv_used
        E_pv_excess = E_pv - E_pv_used
        pv_selfcons.append(E_pv_used)

        # 2. Peak shaving con batteria
        batt_out = 0.0
        if E_residual_load > E_target_step_kWh and soc > 0:
            excess = E_residual_load - E_target_step_kWh
            E_deliverable = min(excess, E_step_limit_kWh, soc * eta_discharge)
            batt_out = E_deliverable
            soc -= E_deliverable / eta_discharge

        # 3. Carica batteria SOLO DA RETE
        batt_in = 0.0
        if E_residual_load < E_target_step_kWh and soc < E_usable:
            deficit = E_target_step_kWh - E_residual_load
            E_storable = min(deficit, E_step_limit_kWh) * eta_charge
            spazio = E_usable - soc
            batt_in = min(E_storable, spazio)
            soc += batt_in

        # 4. Energia rete (import/export)
        E_grid = max(0, E_residual_load - batt_out) + batt_in
        grid_import.append(E_grid)
        grid_export.append(E_pv_excess)

        # Salvataggi
        soc_series.append(soc)
        batt_discharge_out.append(batt_out)
        batt_charge_stored.append(batt_in)

    residual_excess = np.sum(np.maximum(0, np.array(grid_import) - E_target_step_kWh))

    return {
        "soc_series": soc_series,
        "batt_discharge_out_kWh": batt_discharge_out,
        "batt_charge_stored_kWh": batt_charge_stored,
        "grid_import_kWh": grid_import,
        "grid_export_kWh": grid_export,
        "pv_self_consumption_kWh": pv_selfcons,
        "tot_batt_discharge_out_kWh": sum(batt_discharge_out),
        "tot_batt_charge_stored_kWh": sum(batt_charge_stored),
        "tot_grid_import_kWh": sum(grid_import),
        "tot_grid_export_kWh": sum(grid_export),
        "residual_excess": residual_excess
    }

def wrapper_simula_PV(E_step_kWh, E_target_step_kWh, E_nom_kWh, E_step_limit_kWh,
                      eta_charge, eta_discharge, DoD, soc_init_frac, df_mese, simula_funzione):
    """
    Wrapper che passa E_pv_step_kWh se la strategia lo richiede
    """
    import inspect
    args = inspect.signature(simula_funzione).parameters
    if "E_pv_step_kWh" in args:
        E_pv = df_mese["PV_energia"].values
        return simula_funzione(
            E_load_step_kWh=E_step_kWh,
            E_target_step_kWh=E_target_step_kWh,
            E_nom_kWh=E_nom_kWh,
            E_step_limit_kWh=E_step_limit_kWh,
            E_pv_step_kWh=E_pv,
            eta_charge=eta_charge,
            eta_discharge=eta_discharge,
            DoD=DoD,
            soc_init_frac=soc_init_frac
        )
    else:
        return simula_funzione(
            E_step_kWh=E_step_kWh,
            E_target_step_kWh=E_target_step_kWh,
            E_nom_kWh=E_nom_kWh,
            E_step_limit_kWh=E_step_limit_kWh,
            eta_charge=eta_charge,
            eta_discharge=eta_discharge,
            DoD=DoD,
            soc_init_frac=soc_init_frac
        )

print('dayi')
# Trova la capacità minima necessaria (binary search)

def trova_capacity_per_target_carico(
    E_step_kWh,
    E_target_step_kWh,
    E_step_limit_kWh,
    simula_funzione,
    eta_charge=0.95,
    eta_discharge=0.95,
    DoD=0.9,
    tol_kwh=1e-3,
    soc_init_frac=0.5,
    max_iter=40,
):
    """
    Trova la capacità minima necessaria per garantire
    che il residuo sopra soglia sia ≤ tol_kwh,
    usando la funzione di simulazione specificata.
    """
    sum_excess = sum(max(0, e - E_target_step_kWh) for e in E_step_kWh)
    if sum_excess == 0:
        return {"E_nom_kWh": 0.0, "E_usable_kWh": 0.0, "sim": None}

    upper = max(1.0, sum_excess / DoD) * 1.2
    lo, hi = 0.0, upper
    best = None

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        sim = simula_funzione(
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
        sim = simula_funzione(
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



# Dimensionamento per tutti i mesi

def dimensiona_per_tutti_mesi_strategie(
    df_carico,
    soglia_picco_kWh_per_step,
    simula_funzione,
    eta_charge=0.95,
    eta_discharge=0.95,
    DoD=0.9,
    tol_kwh=1e-3,
    soc_init_frac=0.5,
):

    df = df_carico.copy()
    df["Time"] = pd.to_datetime(df["Time"])
    df["Mese"] = df["Time"].dt.to_period("M").astype(str)

    risultati = []

    import inspect
    args_funzione = inspect.signature(simula_funzione).parameters

    for mese, g in df.groupby("Mese"):
        E_step = g["Carico_energia"].values
        E_step_limit = np.max(E_step)

        # Se la funzione richiede PV, verifica la colonna e lunghezza
        if "E_pv_step_kWh" in args_funzione:
            if "PV_energia" not in g.columns:
                raise ValueError(f"Manca colonna PV_energia per la strategia {simula_funzione.__name__}")
            E_pv = g["PV_energia"].values
            if len(E_pv) != len(E_step):
                raise ValueError(f"Mismatch lunghezza carico ({len(E_step)}) e PV ({len(E_pv)}) per il mese {mese}")
            sim = simula_funzione(
                E_load_step_kWh=E_step,
                E_pv_step_kWh=E_pv,
                E_target_step_kWh=soglia_picco_kWh_per_step,
                E_nom_kWh=E_step_limit,
                E_step_limit_kWh=E_step_limit,
                eta_charge=eta_charge,
                eta_discharge=eta_discharge,
                DoD=DoD,
                soc_init_frac=soc_init_frac
            )
        else:
            sim = simula_funzione(
                E_step_kWh=E_step,
                E_target_step_kWh=soglia_picco_kWh_per_step,
                E_nom_kWh=E_step_limit,
                E_step_limit_kWh=E_step_limit,
                eta_charge=eta_charge,
                eta_discharge=eta_discharge,
                DoD=DoD,
                soc_init_frac=soc_init_frac
            )

        # Ora chiamiamo trova_capacity_per_target_carico passando la funzione corretta
        res = trova_capacity_per_target_carico(
            E_step_kWh=E_step,
            E_target_step_kWh=soglia_picco_kWh_per_step,
            E_step_limit_kWh=E_step_limit,
            simula_funzione=lambda *args, **kwargs: wrapper_simula_PV(
                *args,
                df_mese=g,
                simula_funzione=simula_funzione,
                **kwargs
            ),
            eta_charge=eta_charge,
            eta_discharge=eta_discharge,
            DoD=DoD,
            tol_kwh=tol_kwh,
            soc_init_frac=soc_init_frac,
        )

        sim_res = res["sim"]

        # Costruisci aggregati sicuri (se sim_res è None, metti 0)
        risultati.append({
            "Mese": mese,
            "soglia_target_kWh": soglia_picco_kWh_per_step,
            "E_step_limit_KWh": E_step_limit,
            "E_nom_kWh": res["E_nom_kWh"],
            "E_usable_kWh": res["E_usable_kWh"],
            "sum_excess_kWh": sim_res.get("tot_grid_import_kWh", 0) - sim_res.get("tot_batt_discharge_out_kWh", 0) if sim_res else 0,
            "E_dispatched_kWh": sim_res.get("tot_batt_discharge_out_kWh", 0) if sim_res else 0,
            "E_charged_kWh": sim_res.get("tot_batt_charge_stored_kWh", 0) if sim_res else 0,
            "residual_excess_kWh": max(0, sim_res.get("tot_grid_import_kWh", 0) - sim_res.get("tot_batt_discharge_out_kWh", 0)) if sim_res else 0,
        })

    return pd.DataFrame(risultati)




def dimensiona_per_varie_soglie_strategie(
    df_carico,
    lista_soglie,
    strategie,  # dizionario: {"NomeStrategia": funzione_simulazione, ...}
    eta_charge=0.95,
    eta_discharge=0.95,
    DoD=0.9,
    tol_kwh=1e-3,
    soc_init_frac=0.5,
):

    risultati_per_strategia = {nome: [] for nome in strategie.keys()}

    for soglia in lista_soglie:
        for nome_strat, sim_fun in strategie.items():
            df_res = dimensiona_per_tutti_mesi_strategie(
                df_carico,
                soglia_picco_kWh_per_step=soglia,
                simula_funzione=sim_fun,
                eta_charge=eta_charge,
                eta_discharge=eta_discharge,
                DoD=DoD,
                tol_kwh=tol_kwh,
                soc_init_frac=soc_init_frac,
            )
            df_res["Strategia"] = nome_strat
            df_res["Soglia_target"] = soglia
            risultati_per_strategia[nome_strat].append(df_res)

    # Concatena i risultati per ogni strategia in un DataFrame separato
    df_peakshaving = pd.concat(risultati_per_strategia.get("PeakShaving", []), ignore_index=True)
    df_ps_autoconsumo = pd.concat(risultati_per_strategia.get("PS+Autoconsumo", []), ignore_index=True)
    df_ps_gridcharge = pd.concat(risultati_per_strategia.get("PS+GridCharge", []), ignore_index=True)

    return df_peakshaving, df_ps_autoconsumo, df_ps_gridcharge


strategie = {
    "PeakShaving": simula_batteria_carico,
    "PS+Autoconsumo": simula_batteria_peakshaving_autoconsumo,
    "PS+GridCharge": simula_batteria_peakshaving_gridcharge,
}

lista_soglie = [300, 400, 500, 600]

df_peakshaving, df_ps_autoconsumo, df_ps_gridcharge = dimensiona_per_varie_soglie_strategie(
    df_carico,
    lista_soglie,
    strategie,
    eta_charge=0.95,
    eta_discharge=0.95,
    DoD=0.9,
    tol_kwh=1e-3,
    soc_init_frac=0.5,
)

print(df_peakshaving.head())
print(df_ps_autoconsumo.head())
print(df_ps_gridcharge.head())

df_carico["Time"] = pd.to_datetime(df_carico["Time"])
df_carico["Mese"] = df_carico["Time"].dt.to_period("M").astype(str)
mese_scelto = "2024-11"
df_plot = df_carico[df_carico["Mese"] == mese_scelto].copy()



E_step = df_plot["Carico_energia"].values
time = df_plot["Time"].values
E_target_step_kWh = 400#
E_step_limit_kWh = 500

# Simula la batteria
sim_res = simula_batteria_carico(
    E_step_kWh=E_step,
    E_target_step_kWh=E_target_step_kWh,
    E_nom_kWh=E_step_limit_kWh,
    E_step_limit_kWh=E_step_limit_kWh,
    eta_charge=0.95,
    eta_discharge=0.95,
    DoD=0.9,
    soc_init_frac=0.5
)

soc_series = np.array(sim_res["soc_series"])
batt_discharge = np.minimum(E_step - E_target_step_kWh, soc_series * 0.95)
batt_discharge = np.where(batt_discharge > 0, batt_discharge, 0)
batt_charge = np.minimum(E_target_step_kWh - E_step, (E_step_limit_kWh * 0.95))
batt_charge = np.where(batt_charge > 0, batt_charge, 0)

# Plot
plt.figure(figsize=(15,6))
plt.plot(time, E_step, label='Carico', color='blue')
plt.plot(time, [E_target_step_kWh]*len(E_step), '--', label='Soglia', color='red')

# Evidenzia zone di scarica (riduzione picchi)
plt.fill_between(time, E_step - batt_discharge, E_step, color='orange', alpha=0.5, label='Batteria scarica')

# Evidenzia zone di carica (accumulo)
plt.fill_between(time, E_step, E_step + batt_charge, color='green', alpha=0.5, label='Batteria carica')

plt.xlabel("Tempo", fontsize=14)
plt.ylabel("Energia (kWh)", fontsize=14)
plt.title("Andamento carico, soglia e intervento batteria", fontsize=16)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.close()


# Dati
time = df_plot["Time"].values
E_load = df_plot["Carico_energia"].values
E_target_step_kWh = 500

# Simulazione
sim_res = simula_batteria_peakshaving_autoconsumo(
    E_load_step_kWh=E_load,
    E_pv_step_kWh=df_plot["PV_energia"].values,
    E_target_step_kWh=E_target_step_kWh,
    E_nom_kWh=980,
    E_step_limit_kWh=100,
    eta_charge=0.95,
    eta_discharge=0.95,
    DoD=0.9,
    soc_init_frac=0.5
)

batt_discharge = sim_res["batt_discharge_out_kWh"]
batt_charge    = sim_res["batt_charge_stored_kWh"]

# Plot
plt.figure(figsize=(15,6))
plt.plot(time, E_load, label="Carico", color="blue")
plt.plot(time, [E_target_step_kWh]*len(E_load), "--", label="Soglia", color="red")

# Scarica batteria
plt.fill_between(time, E_load - batt_discharge, E_load, color="orange", alpha=0.5, label="Batteria scarica")

# Carica batteria (autoconsumo)
plt.fill_between(time, E_load, E_load + batt_charge, color="green", alpha=0.5, label="Batteria carica")

plt.xlabel("Tempo")
plt.ylabel("Energia (kWh)")
plt.title("PeakShaving + Autoconsumo FV")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.close()

datasets = {
    "PeakShaving": df_peakshaving,
    "PS+Autoconsumo": df_ps_autoconsumo,
    "PS+GridCharge": df_ps_gridcharge
}

# Creiamo una lista per salvare le figure
figure_list = []

for i, (nome, df) in enumerate(datasets.items(), start=1):
    fig = plt.figure(i, figsize=(10, 6))  # figura separata
    figure_list.append(fig)

    # Pivot dei dati
    pivot = df.pivot(index="Mese", columns="Soglia_target", values="E_nom_kWh")

    for soglia in pivot.columns:
        plt.plot(pivot.index, pivot[soglia], marker='o', label=f"{nome} - soglia {soglia} kWh")

    plt.xticks(rotation=45)
    plt.xlabel("Mese", fontsize=14)
    plt.ylabel("Capacità batteria E_nom_kWh", fontsize=14)
    plt.ylim(0, 8000)
    plt.title(f"{nome}", fontsize=16)
    plt.legend()
    plt.tight_layout()

# Mostra tutte le figure contemporaneamente
plt.close()