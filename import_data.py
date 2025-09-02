import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from parameters import PV_param, batt_param

#carica dati pv e datetime (Produzione Fotovoltaica da Synaptic)(kWh)
df_pv = pd.read_csv('data/produzione_pv.csv')
df_pv["Time"] = pd.to_datetime(
    df_pv["Time"].astype(str).str.strip(),
    format="%Y-%m-%d %H:%M:%S",)

#carica dati batterie e datetime (dati batterie da fusione solar) (carica e scarica kW)
df_batt = pd.read_csv('data/batterie.csv')
df_batt["Time"] = pd.to_datetime(
    df_batt["Time"].astype(str).str.strip(),
    format="%Y-%m-%d %H:%M:%S",)

#carica dati prelievo e immissione datetime (kWh)
df_prelievo = pd.read_csv('data/prelievo_immissione.csv')
df_prelievo["Time"] = pd.to_datetime(
    df_prelievo["Time"].astype(str).str.strip(),
    format="%d/%m/%Y %H:%M",)
df_prelievo["Prelievo"] = df_prelievo["Absorption"]
df_prelievo["Immissione"] = df_prelievo["Injection"]


# mette tutto insieme e definisce la data
df = pd.merge(df_pv, df_batt, on="Time")
df = pd.merge(df, df_prelievo, on="Time")
df["Anno"] = df["Time"].dt.year
df["Mese"] = df["Time"].dt.month


# carica prezzi di vendita mensili
prezzi_mensili = pd.DataFrame({
    "Anno": [2022] * 12 + [2023] * 12 + [2024]*12 + [2025]*5,
    "Mese": list(range(1, 13)) * 3 + list(range(1, 6)),
    "Prezzo_vendita": [
        0.058, 0.058, 0.058, 0.058, 0.058, 0.058, 0.058, 0.058, 0.058, 0.058, 0.058, 0.058,
        0.058, 0.058, 0.058, 0.058, 0.058, 0.058, 0.058, 0.092, 0.097, 0.145, 0.514, 0.967,
        0.074, 0.067, 0.059, 0.058, 0.064, 0.080, 0.102, 0.116, 0.087, 0.094, 0.106, 0.114,
        0.129, 0.129, 0.090, 0.053, 0.053
    ],
    "Incentivo": [0.388] * 41
})
df_merge = pd.merge(df, prezzi_mensili, on=["Anno", "Mese"], how="left")

# carica prezzi e costi in bolletta
df_bollette = pd.read_csv('data/dati fascia.csv')
df_bollette["Mese"] = pd.to_datetime(
    df_bollette["Mese"].astype(str).str.strip(),
    format="%Y-%m",)
df_bollette["Anno"] = df_bollette["Mese"].dt.year
df_bollette["Mese"] = df_bollette["Mese"].dt.month
df_merge = pd.merge(df_merge, df_bollette, on=["Anno", "Mese"], how="left")
df_merge = df_merge.drop(columns=['Anno', 'Mese'])

# carica i prezzi nazionali tipo PUN
df_prezzi = pd.read_csv('data/prezzi_energia_2024.csv')
df_merge['CNOR'] = df_prezzi['CNOR']
df_merge['PUN'] = df_prezzi['PUN']
df_merge['fascia'] = df_prezzi['Fascia']


# elabora PV data
df_merge["PV_1"] = df_merge["SolarLab2Sezione1"]
df_merge["PV_1"] = df_merge["PV_1"].clip(upper=PV_param['Sezione_1']['Potenza_nominale_connessione'] * 0.25)
df_merge["PV_2"] = df_merge["SolarLab2Sezione2"]
df_merge["PV_3"] = df_merge["SolarLab2Sezione3"]
df_merge["PV_4"] = df_merge["SolarLab2Sezione4"]
df_merge["PV_4"] = df_merge["PV_4"].clip(upper=PV_param['Sezione_4']['Potenza_nominale_connessione'] * 0.25)
df_merge["PV_aggr"] = df_merge["PV_1"] + df_merge["PV_1"] + df_merge["PV_2"] + df_merge["PV_3"] + df_merge["PV_4"]


# elabora dati betterie
df_merge['P_carica_1'] = df_merge['Potenza_carica_scarica_1'].apply(lambda x: x if x > 0 else 0)
df_merge['P_scarica_1'] = (df_merge['Potenza_carica_scarica_1'].apply(lambda x: -x if x < 0 else 0))
df_merge['P_carica_2'] = df_merge['Potenza_carica_scarica_2'].apply(lambda x: x if x > 0 else 0)
df_merge['P_scarica_2'] = (df_merge['Potenza_carica_scarica_2'].apply(lambda x: -x if x < 0 else 0))
df_merge['P_carica_3'] = df_merge['Potenza_carica_scarica_3'].apply(lambda x: x if x > 0 else 0)
df_merge['P_scarica_3'] = (df_merge['Potenza_carica_scarica_3'].apply(lambda x: -x if x < 0 else 0))
df_merge['P_carica_4'] = df_merge['Potenza_carica_scarica_4'].apply(lambda x: x if x > 0 else 0)
df_merge['P_scarica_4'] = (df_merge['Potenza_carica_scarica_4'].apply(lambda x: -x if x < 0 else 0))
df_merge['P_carica_5'] = df_merge['Potenza_carica_scarica_5'].apply(lambda x: x if x > 0 else 0)
df_merge['P_scarica_5'] = (df_merge['Potenza_carica_scarica_5'].apply(lambda x: -x if x < 0 else 0))
df_merge['P_carica'] = (df_merge['P_carica_1'] + df_merge['P_carica_2'] + df_merge['P_carica_3'] +
                        df_merge['P_carica_4'] + df_merge['P_carica_5'])
df_merge['P_carica'] = df_merge['P_carica'].clip(upper=batt_param['Potenza_nominale_accumulo'])
df_merge['P_scarica'] = (df_merge['P_scarica_1'] +  df_merge['P_scarica_2'] + df_merge['P_scarica_3']
                         + df_merge['P_scarica_4'] + df_merge['P_scarica_5'])
df_merge['P_scarica'] = df_merge['P_scarica'].clip(upper=batt_param['Potenza_nominale_accumulo'])
df_merge['E_carica'] = df_merge['P_carica'] * 0.25
df_merge['E_scarica'] = df_merge['P_scarica'] * 0.25
df_merge['Capacita_scaricabile'] = (df_merge['Capacita_scaricabile_1'] + df_merge['Capacita_scaricabile_2'] +
                                    df_merge['Capacita_scaricabile_3'] + df_merge['Capacita_scaricabile_4'] +
                                    df_merge['Capacit_scaricabile_5'])
df_merge['cap_finale'] = df_merge['Capacita_scaricabile'] + 0.95 * df_merge['P_carica'] * 0.25 - (df_merge['P_scarica'] * 0.25) / 0.95

### BILANCIO PER RICAVARE IL CARICO ###

df_merge['Carico_energia'] = + df_merge['Prelievo'] + df_merge['PV_aggr'] * 0.95 * 0.96 + df_merge['E_scarica'] * 0.99 - df_merge['Immissione'] - df_merge['E_carica'] * 0.99
df_merge['Carico_energia'] = df_merge['Carico_energia'].clip(lower=0)
df_merge.to_csv('dataset_full.csv')
df_merge['net'] = df_merge['Prelievo'] + df_merge['PV_aggr']

# grafico per plottare i dati
#plt.plot(df_merge['Time'], df_merge['Carico_energia'], label='Carico')
plt.plot(df_merge['Time'], df_merge['Prelievo'], label='Prelievo')
#plt.plot(df_merge['Time'], -(df_merge['Immissione']), label='Immissione')
#plt.plot(df_merge['Time'], df_merge['PV_aggr'], label='PV')
#plt.plot(df_merge['Time'], df_merge['P_carica'], label='carica')
#plt.plot(df_merge['Time'], df_merge['P_scarica'], label='scarica')
#plt.plot(df_merge['Time'], df_merge['P_scarica'] + df_merge['PV_aggr_'], label='net')
plt.xlabel("Tempo (15 min)", fontsize=16)
plt.ylabel("Energia (kWh)", fontsize=16)
plt.legend()
plt.xticks(rotation=45, fontsize=16)
plt.tight_layout()
plt.close()


# stampa i valori di media e deviazione standard intero anno
media = df_merge['Carico_energia'].mean()
deviazione_std = df_merge['Carico_energia'].std()
massimo = df_merge['Carico_energia'].max()
print("Valori annuali")
print(f"Media: {media:.2f} kWh")
print(f"Deviazione standard: {deviazione_std:.2f} kWh")
print(f"Massimo: {massimo:.2f} kWh")

# grafico a torta
energia_per_fascia = df_merge.groupby('fascia')['Prelievo'].sum()
percentuali = energia_per_fascia / energia_per_fascia.sum() * 100
labels = ['F1', 'F2', 'F3']
plt.figure(figsize=(6,6))
plt.pie(percentuali.reindex([1,2,3]), labels=labels, autopct='%1.1f%%', colors=['blue', 'orange', 'green'])
plt.close()

# Genera grafici carico per periodo
df_merge['Time'] = pd.to_datetime(df_merge['Time'])
df_merge['AnnoMese'] = df_merge['Time'].dt.to_period('M')

for periodo, gruppo in df_merge.groupby('AnnoMese'):
    plt.figure(figsize=(10, 6))
    plt.plot(gruppo['Time'], gruppo['Carico_energia'], color='steelblue')
    plt.xlabel("Data")
    plt.ylim(0, 600)
    plt.ylabel("Carico (kWh)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'immagini/carico_{periodo}')
    plt.close()

# Genera grafico Istogramma distribuzioni intero anno
bins = np.arange(0, 400, 25)
plt.figure(figsize=(10, 6))
plt.hist(df_merge['Carico_energia'], bins=bins, color='skyblue', edgecolor='black')

plt.xlabel("Energia (kWh)")
plt.ylabel("Numero di intervalli")
plt.xticks(bins)
plt.grid(True)
plt.savefig('immagini/distribuzione.png')

# Genera grafici Istogramma distribuzioni per mese e stampa media e deviazione standard
df_merge['Time'] = pd.to_datetime(df_merge['Time'])
df_merge['AnnoMese'] = df_merge['Time'].dt.to_period('M')

bins = np.arange(0, 401, 25)
df_merge['Costo'] = df_merge['Carico_energia'] * df_merge['CNOR']
for periodo, gruppo in df_merge.groupby('AnnoMese'):
    energia = gruppo['Carico_energia']
    costo = gruppo['Costo'].sum()
    media = energia.mean()
    deviazione_std = energia.std()
    massimo = energia.max()

    print(f"\nMese: {periodo}")
    print(f"Media: {media:.2f} kWh")
    print(f"Deviazione standard: {deviazione_std:.2f} kWh")
    print(f"Massimo: {massimo:.2f} kWh")
    print(f"Costo totale: {costo:.2f} €")

    plt.figure(figsize=(10, 6))
    plt.hist(energia, bins=bins, color='skyblue', edgecolor='black')

    plt.xlabel("Energia (kWh)", fontsize=16)
    plt.ylabel("Numero di intervalli (15 min)", fontsize=16)
    plt.xticks(bins, fontsize=16)
    plt.ylim(0, 800)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'immagini/distribuzione_{periodo}.png')
    plt.close()

# Genera grafici di carico a fasce per mese con pie chart
colors_map = {1: 'tomato', 2: 'gold', 3: 'skyblue'}
legend_elements = [Line2D([0], [0], color=colors_map[i], lw=2, label=f'Fascia {i}') for i in [1, 2, 3]]
for periodo, gruppo in df_merge.groupby('AnnoMese'):

    df_plot = gruppo.sort_values('Time').set_index('Time')
    x = mdates.date2num(df_plot.index.to_pydatetime())
    y = df_plot['Carico_energia'].values
    fascia = df_plot['fascia'].values


    colors = [colors_map[f] for f in fascia]
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    segment_colors = colors[:-1]


    energia_per_fascia = gruppo.groupby('fascia')['Carico_energia'].sum()
    labels = [f"Fascia {int(f)}" for f in energia_per_fascia.index]
    pie_colors = [colors_map[f] for f in energia_per_fascia.index]


    fig, axs = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={'width_ratios': [4, 1]})


    axs[0].add_collection(LineCollection(segments, colors=segment_colors, linewidth=1.8))
    axs[0].set_xlim(x.min(), x.max())
    axs[0].set_ylim(y.min() * 0.95, y.max() * 1.05)
    axs[0].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    fig.autofmt_xdate()
    axs[0].set_xlabel("Tempo", fontsize=12)
    axs[0].set_ylabel("Carico [kWh]", fontsize=12)
    axs[0].legend(handles=legend_elements)

    axs[1].pie(energia_per_fascia, labels=labels, colors=pie_colors, autopct='%1.1f%%')
    axs[1].set_title("Distribuzione Energia per Fascia", fontsize=14)
    plt.tight_layout()
    plt.ylim(0, 600)
    plt.savefig(f"immagini/carico_fasce_{periodo}.png")
    plt.close()

# Genera grafici istogrammi distribuzione a fasce per mese
bins = np.arange(0, 400, 25)
colori = {1: 'tomato', 2: 'gold', 3: 'skyblue'}
fasce = [1, 2, 3]

df_merge['Time'] = pd.to_datetime(df_merge['Time'])
df_merge['AnnoMese'] = df_merge['Time'].dt.to_period('M')

width = bins[1] - bins[0]
df_merge = df_merge.sort_values('Time')
for periodo, gruppo in df_merge.groupby('AnnoMese'):
    energia = gruppo['Carico_energia']

    counts_fasce = []
    for fascia in fasce:
        dati_fascia = gruppo[gruppo['fascia'] == fascia]['Carico_energia']
        counts, _ = np.histogram(dati_fascia, bins=bins)
        counts_fasce.append(counts)

    counts_fasce = np.array(counts_fasce)


    plt.figure(figsize=(10, 6))

    bottom = np.zeros_like(counts_fasce[0])
    for i, fascia in enumerate(fasce):
        plt.bar(bins[:-1], counts_fasce[i], width=width, bottom=bottom,
                color=colori[fascia], edgecolor='black', label=f'Fascia {fascia}')
        bottom += counts_fasce[i]


    plt.xlabel("Energia (kWh)", fontsize=16)
    plt.ylabel("Numero di intervalli", fontsize=16)
    plt.xticks(bins, fontsize=16)
    plt.ylim(0, 800)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'immagini/distribuzione_fasce_{periodo}.png')
    plt.close()


# genera grafico carico per fascie intero anno
df_plot = df_merge.set_index('Time')
x = mdates.date2num(df_plot.index.to_pydatetime())
y = df_plot['Carico_energia'].values
fascia = df_plot['fascia'].values

colors_map = {1: 'tomato', 2: 'gold', 3: 'skyblue'}
colors = [colors_map[f] for f in fascia]

points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
segment_colors = colors[:-1]

lc = LineCollection(segments, colors=segment_colors, linewidth=1.8)

fig, ax = plt.subplots(figsize=(15,6))
ax.add_collection(lc)

ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min() * 0.95, y.max() * 1.05)

ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
fig.autofmt_xdate()

legend_elements = [Line2D([0], [0], color=colors_map[i], lw=2, label=f'Fascia {i}') for i in [1, 2, 3]]
ax.legend(handles=legend_elements)

plt.xlabel("Tempo")
plt.ylabel("Carico [kWh]")
plt.tight_layout()
plt.close()


# genera grafico distribuzione per fasce per un intero anno
bins = np.arange(0, 400, 25)
width = bins[1] - bins[0]
colori = {1: 'tomato', 2: 'gold', 3: 'skyblue'}
fasce = [1, 2, 3]

counts_fasce = []
for fascia in fasce:
    dati_fascia = df_merge[df_merge['fascia'] == fascia]['Carico_energia']
    counts, _ = np.histogram(dati_fascia, bins=bins)
    counts_fasce.append(counts)

counts_fasce = np.array(counts_fasce)

# Plot
plt.figure(figsize=(10, 6))
bottom = np.zeros_like(counts_fasce[0])

for i, fascia in enumerate(fasce):
    plt.bar(bins[:-1], counts_fasce[i], width=width, bottom=bottom,
            color=colori[fascia], edgecolor='black', label=f'Fascia {fascia}')
    bottom += counts_fasce[i]


plt.xlabel("Energia (kWh per intervallo da 15 minuti)")
plt.ylabel("Numero di intervalli")
plt.xticks(bins)
plt.legend(title="Fascia oraria")
plt.grid(True)
plt.tight_layout()
plt.savefig('immagini/distribuzione_fasce.png')

# Calcola energia mensile e risparmiata nel caso dimensionamento batterie
somme_mensili = df_merge.groupby('AnnoMese')['Carico_energia'].sum()
print('Energia complessiva')
print(somme_mensili)
somme_mensili_prelievo = df_merge.groupby('AnnoMese')['Prelievo'].sum()
print('Energia complessiva prelievo')
print(somme_mensili_prelievo)
somma_mensile_250_350 = df_merge.loc[
    (df_merge['Carico_energia'] >= 250) & (df_merge['Carico_energia'] < 350)
].groupby('AnnoMese')['Carico_energia'].sum()

print('energia non prelevata')
print(somma_mensile_250_350)

somma_mensile_250_350_fascia1 = df_merge.loc[
    (df_merge['Carico_energia'] >= 250) &
    (df_merge['Carico_energia'] < 350) &
    (df_merge['fascia'] == 1)
].groupby('AnnoMese')['Carico_energia'].sum()

print('energia non prelevata in fascia1')
print(somma_mensile_250_350_fascia1)

df_merge['Costo'] = df_merge['Carico_energia'] * df_merge['CNOR']
df_merge['Time'] = pd.to_datetime(df_merge['Time'])
df_merge['AnnoMese'] = df_merge['Time'].dt.to_period('M')

costi_mensili = df_merge.groupby('AnnoMese')['Costo'].sum()

print('costi_mensili')
print(costi_mensili)

filtro = (df_merge['Carico_energia'] >= 250) & (df_merge['Carico_energia'] < 350)

# Calcolo dei costi mensili per tutta l'energia in quel range
costi_mensili_250_350 = df_merge.loc[filtro].groupby('AnnoMese')['Costo'].sum()
print("Costi mensili energia tra 250 e 350 kWh:")
print(costi_mensili_250_350)


# Filtra solo la fascia 1
df_fascia1 = df_merge[df_merge['fascia'] == 1]

# Calcola costo per fascia 1
costo_fascia1 = (df_fascia1['Carico_energia'] * df_fascia1['CNOR']).sum()

print(f"Costo energia fascia 1: {costo_fascia1:.2f} €")


