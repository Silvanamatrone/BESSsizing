import pandas as pd
import glob
from datetime import datetime, timedelta
import holidays
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import LineCollection
import matplotlib.dates as mdates
import numpy as np
from matplotlib.lines import Line2D

# Legge tutti i file XML
files = sorted(glob.glob("data/gme/2024*MGPPrezzi.xml"))

all_data = []

for file in files:
    df = pd.read_xml(file, xpath=".//Prezzi")

    df['Data'] = pd.to_datetime(df['Data'], format='%Y%m%d')
    df['Ora'] = df['Ora'].astype(int)

    df['Time'] = df['Data'] + pd.to_timedelta(df['Ora'] - 1, unit='h')

    # Seleziona le colonne che ci interessano
    df_select = df[['Time', 'PUN', 'CNOR']].copy()
    for col in ['PUN', 'CNOR']:
        df_select[col] = df_select[col].str.replace(',', '.').astype(float) / 1000  # €/MWh → €/kWh

    all_data.append(df_select)

df_all = pd.concat(all_data).sort_values("Time").reset_index(drop=True)
df_expanded = df_all.loc[df_all.index.repeat(4)].reset_index(drop=True)
df_expanded['Time'] = df_expanded['Time'] + pd.to_timedelta(df_expanded.groupby('Time').cumcount() * 15, unit='min')
df_expanded = df_expanded.set_index('Time')
italy_holidays = holidays.IT(years=2024)

def fascia_oraria_numeric(ts):
    weekday = ts.weekday()
    hour = ts.hour
    date = ts.date()



    if weekday == 6 or date in italy_holidays:
        return 3

    # Lun-Ven
    if 0 <= weekday <= 4:
        if 8 <= hour < 19:
            return 1
        elif (7 <= hour < 8) or (19 <= hour < 23):
            return 2
        else:
            return 3

    # Sabato
    if weekday == 5:
        if 7 <= hour < 23:
            return 2
        else:
            return 3

df_expanded['Fascia'] = df_expanded.index.to_series().apply(fascia_oraria_numeric)
media_pun_fascia = df_expanded.groupby('Fascia')['PUN'].mean().reindex([1, 2, 3])

# Grafico prezzo medio
sns.set_style("whitegrid")
plt.figure(figsize=(8, 5))
sns.barplot(x=media_pun_fascia.index, y=media_pun_fascia.values, palette="viridis")

plt.title("Prezzo Medio (PUN) per Fascia Oraria", fontsize=14)
plt.ylabel("Prezzo medio [€/kWh]", fontsize=12)
plt.xlabel("Fascia Oraria", fontsize=12)
plt.tight_layout()
plt.show()
# Esporta a CSV
#df_expanded.to_csv("data/prezzi_energia_2024.csv")


#Plot del prezzo per fascia
df_expanded = df_expanded.sort_index()
x = mdates.date2num(df_expanded.index.to_pydatetime())
y = df_expanded['PUN'].values
fascia = df_expanded['Fascia'].values
colors_map = {1: 'blue', 2: 'orange', 3: 'green'}
colors = [colors_map[f] for f in fascia]
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
segment_colors = colors[:-1]

lc = LineCollection(segments, colors=segment_colors, linewidth=2)

fig, ax = plt.subplots(figsize=(15,6))
ax.add_collection(lc)

ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min()*0.95, y.max()*1.05)

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.title("Andamento PUN nel 2024")
plt.xlabel("Tempo")
plt.ylabel("PUN [€/kWh]")
legend_elements = [Line2D([0], [0], color=colors_map[i], lw=2, label=f'Fascia {i}') for i in colors_map]
ax.legend(handles=legend_elements)

plt.tight_layout()
plt.show()
