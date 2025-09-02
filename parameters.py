# Parametri PV

PV_param = {
    "Sezione_1": {
        "Potenza_nominale_connessione": 39.95,   # kW
        "Potenza_picco": 43.05,                  # kWp
        "Potenza_inverter": 40.0,                # kW
        "Corrente_cortocircuito_inverter": 67.0  # A
    },
    "Sezione_2": {
        "Potenza_nominale_connessione": 199.5,   # kW
        "Potenza_picco": 202.5,                  # kWp
        "Potenza_inverter": 199.5,               # kW
        "Corrente_cortocircuito_inverter": 870.0 # A
    },
    "Sezione_3_ante_revamping": {
        "Potenza_picco": 986.0                   # kWp
    },
    "Sezione_3": {
        "Potenza_nominale_connessione": 987.945,   # kW
        "Potenza_picco": 987.945,                  # kWp
        "Potenza_inverter": 1050.0,                # kW
        "Corrente_cortocircuito_inverter": 2025.0  # A
    },
    "Sezione_4_ante_revamping": {
        "Potenza_picco": 1227.2                  # kWp
    },
    "Sezione_4": {
        "Potenza_nominale_connessione": 1237.035,   # kW
        "Potenza_picco": 1238.91,                   # kWp
        "Potenza_inverter": 1265.0,                 # kW
        "Corrente_cortocircuito_inverter": 2587.5   # A
    },
    "Impianto_totale": {
        "Potenza_nominale_connessione": 2464.43,   # kW
        "Potenza_picco": 2472.405,                 # kWp
        "Corrente_cortocircuito_inverter": 5549.5  # A
    }
}


# Parametri batterie

batt_param = {
        "Tensione_nominale": 645.0,          # Vdc
        "Potenza_nominale_accumulo": 500.0,  # kW
        "Potenza_nominale_inverter": 500.0,  # kW
        "Potenza_corto_circuito": 600.0,     # kW
        "Capacita_accumulo_nominale": 983.0, # kWh (196,6 x 5)
        "Cus": 983.0,                        # kWh (196,6 x 5)
        "Potenza_scarica_nominale": 500.0,   # kW
        "Inverter": "n. 5 HUAWEI LUNA2000-100KTL-M1",
        "Accumulo": "n. 5 HUAWEI LUNA2000-200KWH-2H1",
        "Convertitore_PCS": "n.5 HUAWEI LUNA2000-100KTL-M1",
        "Potenza_nominale_sing": 100.0,      # kW
        "Capacita_accumulo_nom_sing": 196.6  # kWh
}