import pandas as pd
from io import StringIO
import numpy as np

# Dati copiati come stringa CSV
data = """Mese,Prezzo F1 (€/kWh),Consumi F1 (kWh),Totale F1 (€),Prezzo F2 (€/kWh),Consumi F2 (kWh),Totale F2 (€),Prezzo F3 (€/kWh),Consumi F3 (kWh),Totale F3 (€),Servizi €/kWh,Servizi kWh,Servizi €,Opzione €/kWh,Opzione kWh,Opzione €,Oneri picco €/kWh,Oneri picco kWh,Oneri picco €,Oneri non picco €/kWh,Oneri non picco kWh,Oneri non picco €,IVA
2024-01,0.119473,126258,15084.41,0.114511,83665,9580.53,0.095215,130398,12415.86,0.007499,340321,2552.4,0.006,327862,1967.17,0.0449,51823,2326.85,0.001639,288498,472.85,10%
2024-02,0.107026,111399,11922.56,0.105269,82560,8691.01,0.084912,138156,11731.16,0.0075,332115,2490.87,0.006,319957,1919.75,0.0449,38868,1745.17,0.001639,293247,480.63,10%
2024-03,0.106299,94614,10057.34,0.109493,68387,7487.93,0.092581,104257,9652.2,0.007499,267258,2004.42,0.006,257473,1544.84,0.0449,13777,618.59,0.001639,253481,415.46,10%
2024-04,0.106869,43265,4623.68,0.122669,41084,5039.72,0.090353,87729,7926.62,0.006868,172078,1182,0.006,165778,994.67,, ,0.002767,172078,476.14,10%
2024-05,0.109764,47026,5161.76,0.13082,33370,4365.46,0.098367,70413,6926.3,0.006869,150809,1035.91,0.006,145287,871.72,, ,0.002767,150809,417.29,10%
2024-06,0.115105,33642,3872.37,0.138876,26326,3656.05,0.108526,58204,6316.62,0.006869,118172,811.73,0.006,113845,683.07,0.0449,8254,370.6,0.002767,109918,304.14,10%
2024-07,0.120397,31443,3785.66,0.135038,29363,3965.12,0.111141,66763,7420.13,0.006383,127569,814.4,0.006,122899,737.39,0.0449,14204,637.76,0.002844,113365,322.41,10%
2024-08,0.144581,16189,2340.63,0.165312,25036,4138.76,0.128448,49841,6401.98,0.006384,91066,581.38,0.006,87731,526.39,0.0449,250,11.23,0.002844,90816,258.28,10%
2024-09,0.136285,89716,12226.99,0.151895,50273,7636.23,0.115766,87809,10165.29,0.006383,227798,1454.25,0.006,219459,1316.75,, ,0.002844,227798,647.86,10%
2024-10,0.135989,121071,16464.34,0.144056,63743,9182.59,0.110728,108507,12014.8,0.008226,293321,2413.15,0.006,282583,1695.49,, ,0.002995,293321,878.5,10%
2024-11,0.157294,126127,19839.07,0.151223,75634,11437.62,0.124773,126683,15806.63,0.008227,328444,2702.13,0.006,316420,1898.52,, ,0.002995,328444,983.69,10%
2024-12,0.16997,138311,23508.69,0.160292,78640,12605.35,0.12056,144155,17379.28,0.008226,361106,2970.81,0.006,347887,2087.33,0.0449,51647,2318.95,0.002995,309459,926.83,10%"""


# Creazione del DataFrame
df = pd.read_csv(StringIO(data), sep=",")
colonne_numeriche = [
    'Prezzo F1 (€/kWh)', 'Consumi F1 (kWh)', 'Totale F1 (€)',
    'Prezzo F2 (€/kWh)', 'Consumi F2 (kWh)', 'Totale F2 (€)',
    'Prezzo F3 (€/kWh)', 'Consumi F3 (kWh)', 'Totale F3 (€)',
    'Servizi €/kWh','Servizi kWh','Servizi €',
    'Opzione €/kWh','Opzione kWh','Opzione €',
    'Oneri picco €/kWh','Oneri picco kWh','Oneri picco €',
    'Oneri non picco €/kWh','Oneri non picco kWh','Oneri non picco €'
]

for col in colonne_numeriche:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
df['Oneri picco kWh'] = df['Oneri picco kWh'].fillna(0)
df['Oneri picco kWh'] = df['Oneri picco kWh'].astype(float)

df_prelievo = pd.read_csv('data/prelievo_immissione.csv')
df_prelievo["Time"] = pd.to_datetime(
    df_prelievo["Time"].astype(str).str.strip(),
    format="%d/%m/%Y %H:%M",)
df_prelievo["Prelievo"] = df_prelievo["Absorption"]
df_prelievo["Immissione"] = df_prelievo["Injection"]

# Mostra le prime righe per verifica
print(df_prelievo.head())


def calcola_picco_mensile(df_prelievo, df_bolletta):
    """
    df_prelievo: DataFrame con colonne 'Time' e 'Prelievo' (consumi quartorari)
    df_bolletta: DataFrame bolletta con colonne 'Mese' e 'Oneri picco kWh'

    Restituisce df_prelievo con colonne aggiuntive:
    - Picco: True se il quartoro è di picco
    - Soglia_picco: soglia minima di picco per il mese
    """
    # Assicuriamoci che la colonna Time sia datetime
    df_prelievo['Time'] = pd.to_datetime(df_prelievo['Time'])
    df_prelievo['Mese'] = df_prelievo['Time'].dt.to_period('M').astype(str)

    risultati = []

    for mese, group in df_prelievo.groupby('Mese'):
        # Prendi i kWh di picco dalla bolletta
        kwh_picco = df_bolletta.loc[df_bolletta['Mese'] == mese, 'Oneri picco kWh'].values[0]

        consumi = group['Prelievo'].values
        total_mese = consumi.sum()

        # Controllo robusto: se kWh di picco > consumo totale, soglia = massimo consumo
        if kwh_picco >= total_mese:
            soglia_picco = consumi.max()
        else:
            # Ordiniamo consumi dal più alto al più basso
            sorted_consumi = np.sort(consumi)[::-1]
            cumsum = np.cumsum(sorted_consumi)

            # Troviamo il quartoro minimo che rientra nei kWh di picco
            idx = np.searchsorted(cumsum, kwh_picco, side='left')
            idx = min(idx, len(sorted_consumi) - 1)
            soglia_picco = sorted_consumi[idx]

        # Marca i quartori come picco o non picco
        group['Picco'] = group['Prelievo'] >= soglia_picco
        group['Soglia_picco'] = soglia_picco

        risultati.append(group)

        # Stampa informazioni mese per mese
        print(f"{mese}: soglia di picco = {soglia_picco:.3f} kWh, quartori di picco = {group['Picco'].sum()}")

    # Ricongiunge tutti i mesi
    df_risultato = pd.concat(risultati).sort_values('Time').reset_index(drop=True)
    return df_risultato

df_prelievo_picco = calcola_picco_mensile(df_prelievo, df)

# Riassunto mese per mese
riassunto_picco = df_prelievo_picco.groupby('Mese').agg(
    Soglia_picco=('Soglia_picco', 'first'),
    Quartori_picco=('Picco', 'sum'),
    Quartori_totali=('Picco', 'count')
).reset_index()

print(riassunto_picco)