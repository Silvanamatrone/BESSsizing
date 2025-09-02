import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from parameters import batt_param
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


# Carico dataset
df_dati = pd.read_csv("dataset_full.csv")
df_dati["Time"] = pd.to_datetime(
    df_dati["Time"].astype(str).str.strip(),
    format="%Y-%m-%d %H:%M:%S",)
df_dati = df_dati[['Time', 'Prelievo', 'Immissione', 'Prezzo F1', 'Prezzo F2', 'Prezzo F3',
                   'Totale F1', 'Totale F2', 'Totale F3', 'PV_aggr', 'Carico_energia', 'fascia']]

delta_t = 0.25
DoD = 0.9
eta_charge = 0.95
eta_discharge = 0.95
costo_unitario_kWh = 300  # €/kWh
perc_manutenzione = 0.02  # 2% annuo

# Liste di parametri da testare
n_BESS_list = [5, 10, 20, 30, 40]
soglia_percentuali = [0.7, 0.8, 0.9]

# Qui raccogliamo i risultati
risultati = []

for soglia_percentuale in soglia_percentuali:
    for n_BESS in n_BESS_list:
        # Parametri batteria
        P_batt_max = batt_param["Potenza_nominale_sing"] * n_BESS
        E_batt_max = P_batt_max * delta_t
        E_nom_kWh = batt_param["Capacita_accumulo_nom_sing"] * n_BESS
        E_usable = E_nom_kWh * DoD
        SOC_init = 0.5 * E_usable

        SOC_series = []
        soc = SOC_init
        E_carica_list = []
        E_scarica_list = []

        # Copia dati per simulazione
        df = df_dati.copy()

        df['Data'] = df['Time'].dt.date
        df['Picco_giornaliero'] = df.groupby('Data')['Carico_energia'].transform('max')
        df['Soglia_picco'] = df['Picco_giornaliero'] * soglia_percentuale
        df['Picco_F1'] = ((df['fascia'] == 1) & (df['Carico_energia'] >= df['Soglia_picco'])).astype(int)
        df['BESS_carica'] = 0.0
        df.loc[df['Picco_F1'] == 1, 'BESS_scarica'] = df.loc[df['Picco_F1'] == 1, 'Carico_energia'].clip(upper=E_batt_max)

        # Simulazione SOC dinamico
        for i, row in df.iterrows():
            E_carica = 0.0
            E_scarica = 0.0
            if row['Picco_F1'] == 1:
                E_scarica = min(row['BESS_scarica'], soc * eta_discharge)
                soc -= E_scarica / eta_discharge
            elif row['fascia'] == 3:
                spazio = E_usable - soc
                E_carica = min(E_batt_max * eta_charge, spazio)
                soc += E_carica

            SOC_series.append(soc)
            E_carica_list.append(E_carica)
            E_scarica_list.append(E_scarica)

        df['SOC'] = SOC_series
        df['E_carica'] = E_carica_list
        df['E_scarica'] = E_scarica_list
        df['Carico_effettivo'] = df['Carico_energia'] - df['E_scarica'] + df['E_carica']

        # Carichi per fascia
        df['Carico_F1'] = np.where(df['fascia'] == 1, df['Carico_energia'], 0)
        df['Carico_F2'] = np.where(df['fascia'] == 2, df['Carico_energia'], 0)
        df['Carico_F3'] = np.where(df['fascia'] == 3, df['Carico_energia'], 0)

        df['Carico_effettivo_F1'] = df['Carico_F1'] - df['E_scarica']
        df['Carico_effettivo_F2'] = df['Carico_F2']
        df['Carico_effettivo_F3'] = df['Carico_F3'] + df['E_carica']

        # Costi
        df['Costo_F1'] = df['Carico_F1'] * df['Prezzo F1']
        df['Costo_F2'] = df['Carico_F2'] * df['Prezzo F2']
        df['Costo_F3'] = df['Carico_F3'] * df['Prezzo F3']

        df['Costo_effettivo_F1'] = df['Carico_effettivo_F1'] * df['Prezzo F1']
        df['Costo_effettivo_F2'] = df['Carico_effettivo_F2'] * df['Prezzo F2']
        df['Costo_effettivo_F3'] = df['Carico_effettivo_F3'] * df['Prezzo F3']

        df['Costo_totale'] = df['Costo_F1'] + df['Costo_F2'] + df['Costo_F3']
        df['Costo_totale_effettivo'] = df['Costo_effettivo_F1'] + df['Costo_effettivo_F2'] + df['Costo_effettivo_F3']
        df['Delta_Costo_totale'] = df['Costo_totale'] - df['Costo_totale_effettivo']


        # Risultati annuali
        carico_F1 = df['Carico_F1'].sum()
        risparmio_carico_F1 = (df['Carico_F1'] - df['Carico_effettivo_F1']).sum()
        costo_totale = df['Costo_totale'].sum()
        risparmio_costo_totale = df['Delta_Costo_totale'].sum()
        CAPEX = E_nom_kWh * costo_unitario_kWh
        OPEX_annuo = CAPEX * perc_manutenzione
        costo_batteria = CAPEX + OPEX_annuo

        # Salva risultati in lista
        risultati.append({
            "Soglia [%]": soglia_percentuale * 100,
            "n_BESS": n_BESS,
            "Carico_F1 [kWh]": carico_F1,
            "Risparmio_Carico_F1 [kWh]": risparmio_carico_F1,
            "Costo_Annuale [€]": costo_totale,
            "Risparmio_Costo_Annuale [€]": risparmio_costo_totale,
            "Costo_batteria [€]": costo_batteria
        })

# Converto in DataFrame
df_risultati = pd.DataFrame(risultati)
print(df_risultati)


# Lista delle soglie e dei n_BESS testati
soglie = df_risultati["Soglia [%]"].unique()
n_bess_list = df_risultati["n_BESS"].unique()

# --- 1. Risparmio vs n_BESS per diverse soglie ---
plt.figure(figsize=(10,6))
for soglia in soglie:
    subset = df_risultati[df_risultati["Soglia [%]"] == soglia]
    plt.plot(subset["n_BESS"], subset["Risparmio_Costo_Annuale [€]"],
             marker="o", label=f"Soglia {int(soglia*100)}%")
plt.xlabel("Numero di Batterie (n_BESS)")
plt.ylabel("Risparmio Costo Annuale [€]")
plt.title("Andamento Risparmio vs Numero Batterie per diverse Soglie")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 2. Risparmio vs Soglia per diversi n_BESS ---
plt.figure(figsize=(10,6))
for n_bess in n_bess_list:
    subset = df_risultati[df_risultati["n_BESS"] == n_bess]
    plt.plot(subset["Soglia [%]"]*100, subset["Risparmio_Costo_Annuale [€]"],
             marker="s", label=f"{n_bess} BESS")
plt.xlabel("Soglia [%]")
plt.ylabel("Risparmio Costo Annuale [€]")
plt.title("Andamento Risparmio vs Soglia per diverse Taglie di Batteria")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



df_dati = pd.read_csv('dataset_full.csv')
df_dati["Time"] = pd.to_datetime(
    df_dati["Time"].astype(str).str.strip(),
    format="%Y-%m-%d %H:%M:%S",)

df_dati = df_dati[['Time', 'Prelievo', 'Immissione', 'Prezzo F1', 'Prezzo F2', 'Prezzo F3', 'Totale F1', 'Totale F2', 'Totale F3', 'PV_aggr', 'Carico_energia', 'fascia']]

soglia_percentuale = 0.8
n_BESS = 5
P_batt_max = batt_param["Potenza_nominale_sing"] * n_BESS
delta_t = 0.25
E_batt_max = P_batt_max * delta_t
E_nom_kWh = batt_param["Capacita_accumulo_nom_sing"] * n_BESS
DoD = 0.9
E_usable = E_nom_kWh * DoD
eta_charge = 0.95
eta_discharge = 0.95
SOC_init = 0.5 * E_usable

SOC_series = []
soc = SOC_init
E_carica_list = []
E_scarica_list = []


df_dati['Data'] = df_dati['Time'].dt.date
df_dati['Picco_giornaliero'] = df_dati.groupby('Data')['Carico_energia'].transform('max')
df_dati['Soglia_picco'] = df_dati['Picco_giornaliero'] * soglia_percentuale
df_dati['Picco'] = (df_dati['Carico_energia'] >= df_dati['Soglia_picco']).astype(int)
df_dati['Picco_F1'] = ((df_dati['fascia'] == 1) & (df_dati['Carico_energia'] >= df_dati['Soglia_picco'])).astype(int)
df_dati['BESS_carica'] = 0.0
df_dati.loc[df_dati['Picco_F1'] == 1, 'BESS_scarica'] = df_dati.loc[df_dati['Picco_F1'] == 1, 'Carico_energia'].clip(
    upper=E_batt_max)

for i, row in df_dati.iterrows():
    E_carica = 0.0
    E_scarica = 0.0
    # Scarica se picco in F1
    if row['Picco_F1'] == 1:
        E_scarica = min(row['BESS_scarica'], soc * eta_discharge)
        soc -= E_scarica / eta_discharge
    # Carica se in F3
    elif row['fascia'] == 3:
        spazio = E_usable - soc
        E_carica = min(E_batt_max * eta_charge, spazio)
        soc += E_carica

    SOC_series.append(soc)
    E_carica_list.append(E_carica)
    E_scarica_list.append(E_scarica)

df_dati['SOC'] = SOC_series
df_dati['E_carica'] = E_carica_list
df_dati['E_scarica'] = E_scarica_list

df_dati['Carico_effettivo'] = df_dati['Carico_energia'] - df_dati['E_scarica'] + df_dati['E_carica']

# Carico originale per fascia
df_dati['Carico_F1'] = np.where(df_dati['fascia'] == 1, df_dati['Carico_energia'], 0)
df_dati['Carico_F2'] = np.where(df_dati['fascia'] == 2, df_dati['Carico_energia'], 0)
df_dati['Carico_F3'] = np.where(df_dati['fascia'] == 3, df_dati['Carico_energia'], 0)

# Carico effettivo per fascia considerando la batteria
df_dati['Carico_effettivo_F1'] = df_dati['Carico_F1'] - df_dati['E_scarica']
df_dati['Carico_effettivo_F2'] = df_dati['Carico_F2']  # nessuna variazione
df_dati['Carico_effettivo_F3'] = df_dati['Carico_F3'] + df_dati['E_carica']

# Costi per carico originale
df_dati['Costo_F1'] = df_dati['Carico_F1'] * df_dati['Prezzo F1']
df_dati['Costo_F2'] = df_dati['Carico_F2'] * df_dati['Prezzo F2']
df_dati['Costo_F3'] = df_dati['Carico_F3'] * df_dati['Prezzo F3']

# Costi per carico effettivo (dopo intervento BESS)
df_dati['Costo_effettivo_F1'] = df_dati['Carico_effettivo_F1'] * df_dati['Prezzo F1']
df_dati['Costo_effettivo_F2'] = df_dati['Carico_effettivo_F2'] * df_dati['Prezzo F2']
df_dati['Costo_effettivo_F3'] = df_dati['Carico_effettivo_F3'] * df_dati['Prezzo F3']

# Costo totale originale e effettivo
df_dati['Costo_totale'] = df_dati['Costo_F1'] + df_dati['Costo_F2'] + df_dati['Costo_F3']
df_dati['Costo_totale_effettivo'] = df_dati['Costo_effettivo_F1'] + df_dati['Costo_effettivo_F2'] + df_dati['Costo_effettivo_F3']
df_dati['Delta_Costo_totale'] = df_dati['Costo_totale'] - df_dati['Costo_totale_effettivo']
df_dati['Delta_Carico_totale'] = df_dati['Carico_F1'] + df_dati['Carico_F2'] + df_dati['Carico_F3'] \
                               - (df_dati['Carico_effettivo_F1'] + df_dati['Carico_effettivo_F2'] + df_dati['Carico_effettivo_F3'])

# Differenze tra carico originale e carico effettivo
df_dati['Delta_F1'] = df_dati['Carico_F1'] - df_dati['Carico_effettivo_F1']
df_dati['Delta_F2'] = df_dati['Carico_F2'] - df_dati['Carico_effettivo_F2']
df_dati['Delta_F3'] = df_dati['Carico_F3'] - df_dati['Carico_effettivo_F3']

# Differenze tra costo originale e costo effettivo
df_dati['Delta_Costo_F1'] = df_dati['Costo_F1'] - df_dati['Costo_effettivo_F1']
df_dati['Delta_Costo_F2'] = df_dati['Costo_F2'] - df_dati['Costo_effettivo_F2']
df_dati['Delta_Costo_F3'] = df_dati['Costo_F3'] - df_dati['Costo_effettivo_F3']

# Risparmio totale di carico in F1 [kWh]
carico_F1 = df_dati['Carico_F1'].sum()
risparmio_carico_F1 = df_dati['Delta_F1'].sum()

# Risparmio totale di costo annuale [€]
costo_totale = df_dati['Costo_totale'].sum()
risparmio_costo_totale = df_dati['Delta_Costo_totale'].sum()

print(f"Totale carico in F1: {carico_F1:.2f} kWh")
print(f"Risparmio totale di carico in F1: {risparmio_carico_F1:.2f} kWh")
print(f"Costo annuale: {costo_totale:.2f} €")
print(f"Risparmio totale di costo annuale: {risparmio_costo_totale:.2f} €")



df_dati['AnnoMese'] = df_dati['Time'].dt.to_period('M')

colors_map = {1: 'tomato', 2: 'gold', 3: 'skyblue'}
legend_elements = [Line2D([0], [0], color=colors_map[i], lw=2, label=f'Fascia {i}') for i in [1, 2, 3]]

for periodo, gruppo in df_dati.groupby('AnnoMese'):
    df_plot = gruppo.sort_values('Time').set_index('Time')
    x = mdates.date2num(df_plot.index.to_pydatetime())
    y = df_plot['Carico_energia'].values
    fascia = df_plot['fascia'].values

    # Line segments colorati per fascia
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    segment_colors = [colors_map[f] for f in fascia[:-1]]

    fig, ax = plt.subplots(figsize=(15,6))

    # Plot line collection per fasce
    ax.add_collection(LineCollection(segments, colors=segment_colors, linewidth=1.8))

    # Plot soglia giornaliera
    if 'Soglia_picco' in df_plot.columns:
        ax.plot(df_plot.index, df_plot['Soglia_picco'], color='orange', linestyle='--', label='Soglia giornaliera')
        soglia = df_plot['Soglia_picco'].values
    else:
        soglia = np.zeros(len(df_plot))

    # Energia scaricata sopra la soglia (in alto)
    if 'E_scarica' in df_plot.columns:
        ax.fill_between(df_plot.index, soglia, soglia + df_plot['E_scarica'],
                        color='red', alpha=0.3, label='Energia scaricata')

    # Energia caricata sotto la soglia (in basso)
    if 'E_carica' in df_plot.columns:
        ax.fill_between(df_plot.index, soglia, soglia - df_plot['E_carica'],
                        color='green', alpha=0.3, label='Energia caricata')

    # Asse e legenda
    ax.set_xlim(df_plot.index.min(), df_plot.index.max())
    ax.set_ylim(0, y.max()*1.2)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    fig.autofmt_xdate()
    ax.set_xlabel("Tempo")
    ax.set_ylabel("Carico [kWh]")
    ax.legend(handles=legend_elements + ax.get_legend_handles_labels()[0])
    ax.set_title(f"Andamento carico e BESS - {periodo}")

    plt.tight_layout()
    plt.close()

df_dati['AnnoMese'] = df_dati['Time'].dt.to_period('M')

colors_map = {1: 'tomato', 2: 'gold', 3: 'skyblue'}
legend_elements = [Line2D([0], [0], color=colors_map[i], lw=2, label=f'Fascia {i}') for i in [1, 2, 3]]

for periodo, gruppo in df_dati.groupby('AnnoMese'):
    df_plot = gruppo.sort_values('Time').set_index('Time')
    x = mdates.date2num(df_plot.index.to_pydatetime())
    y = df_plot['Carico_energia'].values
    fascia = df_plot['fascia'].values

    # Line segments colorati per fascia
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    segment_colors = [colors_map[f] for f in fascia[:-1]]

    fig, ax = plt.subplots(figsize=(15,6))

    # Plot line collection per fasce
    ax.add_collection(LineCollection(segments, colors=segment_colors, linewidth=1.8))

    # Plot soglia giornaliera
    if 'Soglia_picco' in df_plot.columns:
        ax.plot(df_plot.index, df_plot['Soglia_picco'], color='orange', linestyle='--', label='Soglia giornaliera')
        soglia = df_plot['Soglia_picco'].values
    else:
        soglia = np.zeros(len(df_plot))

    # Energia scaricata sopra la soglia (in alto)
    if 'E_scarica' in df_plot.columns:
        ax.fill_between(df_plot.index, soglia, soglia + df_plot['E_scarica'],
                        color='red', alpha=0.3, label='Energia scaricata')

    # Energia caricata sotto la soglia (in basso)
    if 'E_carica' in df_plot.columns:
        ax.fill_between(df_plot.index, soglia, soglia - df_plot['E_carica'],
                        color='green', alpha=0.3, label='Energia caricata')

    # Carico effettivo = carico reale - scarica + carica
    if 'E_scarica' in df_plot.columns and 'E_carica' in df_plot.columns:
        df_plot['Carico_effettivo'] = df_plot['Carico_energia'] - df_plot['E_scarica'] + df_plot['E_carica']
        ax.plot(df_plot.index, df_plot['Carico_effettivo'], color='green', linestyle='-', linewidth=2, label='Carico effettivo BESS')

    # Asse e legenda
    ax.set_xlim(df_plot.index.min(), df_plot.index.max())
    ax.set_ylim(0, max(y)*1.2)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    fig.autofmt_xdate()
    ax.set_xlabel("Tempo")
    ax.set_ylabel("Carico [kWh]")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=legend_elements + handles, fontsize=12)
    ax.set_title(f"Andamento carico e BESS - {periodo}")

    plt.tight_layout()
    plt.show()






