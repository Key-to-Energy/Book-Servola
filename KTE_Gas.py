import KTE_artesian as kta
from KTE_time import *
import pandas as pd
import numpy as np
import math
import gas_settings as gas_sett
import datetime as dt
import os
from azure.storage.blob import BlobServiceClient


# Funzione che carica gl'input generici dall'omonimo file
def input_loader():
    df = readCSVFromAzureBlobStorage('https://devmodelingkte.blob.core.windows.net/', 
                                     'ChVeQpT+0v8ylm/xP+PmJVDGnbXMQ6pENrSm171reT4lZTUO9SowNsE8ikbSBvxW8qsjdLBi/FNQblT/uDQeEw==', 
                                     'book-servola', 
                                     '/', 
                                     'Input Book GEFS.xlsx',
                                     index_col=None,
                                     sheet_name="input generici")
    df = df.drop(['Elimina_0', 'Elimina_1'], axis=1)
    return df


# Funzione che carica gli input riguardanti l'energia spot dall'omonimo foglio
def load_programma():
    df = readCSVFromAzureBlobStorage('https://devmodelingkte.blob.core.windows.net/', 
                                     'ChVeQpT+0v8ylm/xP+PmJVDGnbXMQ6pENrSm171reT4lZTUO9SowNsE8ikbSBvxW8qsjdLBi/FNQblT/uDQeEw==', 
                                     'book-servola', 
                                     '/', 
                                     'Input Book GEFS.xlsx', 
                                     index_col=None, 
                                     sheet_name="Energia spot")
    return df


# Funzione che carica le valiabili del gas legate a contratti e simili
def input_variabili_gas_loader():
    df = readCSVFromAzureBlobStorage('https://devmodelingkte.blob.core.windows.net/', 
                                     'ChVeQpT+0v8ylm/xP+PmJVDGnbXMQ6pENrSm171reT4lZTUO9SowNsE8ikbSBvxW8qsjdLBi/FNQblT/uDQeEw==', 
                                     'book-servola', 
                                     '/', 
                                     'Input Book GEFS.xlsx', 
                                     index_col=None, 
                                     sheet_name="variabili gas", 
                                     usecols='D:H', 
                                     skiprows=[1],
                                     header=1)
    return df


# Funzione che carica i volumi della caldaia
def input_caldaia():
    df = pd.read_excel('Input Book GEFS.xlsx', sheet_name="Caldaia", header=0)
    return df


def get_prezzi_heren():
    x = kta.get_artesian_data_actual([100217365], '2020-12-01', dt.datetime.today().strftime('%Y-%m-%d'), 'd')
    x = kta.format_artesian_data(x)
    x = x.rename(columns={'Spot price gas PSV': 'Prezzo Heren'})
    return x


# Funzione per il calcolo dei M3 utili
def make_m3(kwh_giornalieri):
    if math.isnan(kwh_giornalieri):
        return 0
    else:
        return kwh_giornalieri * 0.09458277


# Funzione che selegiona il miglior gas da tenere in considerazione per i calcoli tra, quantità messa a verbale da SNAM,
# il consuntivo, o la nomina
def make_best_gas_column(verbale_snam, allocato, m3):
    if math.isnan(verbale_snam):
        if math.isnan(allocato):
            if math.isnan(m3):
                return 0
            else:
                return m3
        else:
            return allocato
    else:
        return verbale_snam


# Funzione per il calcolo corretto della colonna PCS, per i valori consuntivi userà il consuntivo, per i calori futuri
# userà la media del mese m-1
def make_pcs(pcs, all_pcs):
    mean_pcs = pd.DataFrame()
    mean_pcs['PCSx'] = all_pcs['PCSx']
    mean_pcs['date'] = all_pcs['date']
    mean_pcs['Datetime'] = pd.to_datetime(mean_pcs['date'], utc=True)
    mean_pcs = mean_pcs.set_index('Datetime')
    mean_pcs = mean_pcs.resample('M', label='right').mean()
    mese = dt.datetime.now().month

    if math.isnan(pcs):
        return mean_pcs['PCSx'].loc[mean_pcs.index.month == (mese - 1)].to_list().pop()
    else:
        return pcs


def make_volume_38100_column(pcs, miglior_gas):
    if math.isnan((pcs * miglior_gas) / 10.57275):
        return 0
    else:
        return round((pcs * miglior_gas) / 10.57275)


def make_delta_sbilanciamento_vendita(miglior_gas, m3, tolleranza):
    if miglior_gas < m3:
        return min(0, round(miglior_gas + tolleranza - m3))
    return np.nan


def make_delta_sbilanciamento_acquisto(miglior_gas, m3, tolleranza):
    if miglior_gas > m3:
        return max(0, round(miglior_gas - tolleranza - m3))
    return np.nan


def make_tollerance_columns(data):
    if data.dayofweek > 4:
        return 113000
    else:
        return 56500


def make_heren_price_column(psv):
    return (psv * 10.5833) / 10


def make_delta_tolleranza_acquisto(pcs, miglior_gas, m3, tolleranza):
    consumo = (pcs * miglior_gas) / 10.57275
    if consumo > m3:
        sbilanciamento = consumo - m3 - tolleranza
        if sbilanciamento > 0:
            return tolleranza
        else:
            return consumo - m3
    return np.nan


def make_delta_tolleranza_vendita(pcs, miglior_gas, m3, tolleranza):
    consumo = (pcs * miglior_gas) / 10.57275
    if consumo < m3:
        sbilanciamento = m3 - tolleranza - consumo
        if sbilanciamento > 0:
            return tolleranza
        else:
            return - (- sbilanciamento - tolleranza)
    return np.nan


def make_prezzo_smc_38100(psv):
    return psv * 10.5833 / 10


def make_prezzo_tolleranza(prezzo, giorno):
    if giorno.dayofweek > 4:
        return prezzo * 0.03
    else:
        return prezzo * 0.02


def make_prezzo_sbilanciamento_smc_cent(prezzo):
    return (prezzo * 10.5833) / 10


def make_prezzo_sbilanciamento_acquisto(sap, max_snam):
    return max([(sap + 0.108), max_snam])


def make_prezzo_sbilanciamento_vendita(sap, min_snam):
    return min([(sap - 0.108), min_snam])


def make_variabili_gas(giorno, df_input_variabili):
    return df_input_variabili['Veriabili senza fee c€/smc'].loc[
        (df_input_variabili['Mese'].dt.month == giorno.month) &
        (df_input_variabili['Mese'].dt.year == giorno.year)]


def make_fee_contratto_smc_cent(giorno, df_input_variabili):
    return df_input_variabili['Fee c€/smc 38,1'].loc[
        (df_input_variabili['Mese'].dt.month == giorno.month) &
        (df_input_variabili['Mese'].dt.year == giorno.year)]


def make_fee_contratto_smc_cent_book(df, df_input_variabili):
    fee_list = list()
    for index, row in df.iterrows():
        try:
            fee_list.append((
                                    make_fee_contratto_smc_cent(row['date'], df_input_variabili).array[0]
                                    * row['PCS']) / 10.57275)
        except:
            fee_list.append(np.nan)

    df['Fee c€/smc 38,1'] = fee_list
    return df


def make_variabili_gas_book(df, df_input_variabili):
    fee_list = list()
    for index, row in df.iterrows():
        try:
            fee_list.append(make_variabili_gas(row['date'], df_input_variabili).array[0])
        except:
            fee_list.append(np.nan)

    df['Variabili gas'] = fee_list
    return df


def make_heren_price(date, df_heren):
    if pd.isnull(date):
        return np.nan
    else:
        tmp = df_heren['Prezzo Heren'].loc[df_heren['Date'] == pd.Timestamp(day=date.day,
                                           year=date.year,
                                           month=date.month,
                                           hour=date.hour,
                                           ).tz_localize('CET')]
        if len(list(tmp.values)) == 0:
            return np.nan
        else:
            return list(tmp.values)[0]


def get_prezzi_sbilanciamento():
    dati_spot_gas = kta.get_artesian_data_actual_daily([100002806, 100002804, 100002805], '2021-12-01', '2024-12-31')
    df_spot_gas = kta.format_artesian_data(dati_spot_gas)
    df_spot_gas['Prezzo Sbil acquisto'] = df_spot_gas.apply(
        lambda x: make_prezzo_sbilanciamento_acquisto(x['GAS_EsitiPrezzoSbilanciamentoMedioPonderato'],
                                                      x['GAS_EsitiPrezzoSbilanciamentoMassimoSRG']), axis=1)
    df_spot_gas['Prezzo Sbil vendita'] = df_spot_gas.apply(
        lambda x: make_prezzo_sbilanciamento_vendita(x['GAS_EsitiPrezzoSbilanciamentoMedioPonderato'],
                                                     x['GAS_EsitiPrezzoSbilanciamentoMinimoSRG']), axis=1)

    return df_spot_gas


def make_sbilancio(delta_sbil_a, prezzo_sbil_a, prezzo, delta_sbil_v, prezzo_sbil_v, fee_contratto):
    if math.isnan(delta_sbil_a):
        delta_sbil_a = 0

    if math.isnan(prezzo_sbil_a):
        prezzo_sbil_a = 0

    if math.isnan(prezzo):
        prezzo = 0

    if math.isnan(delta_sbil_v):
        delta_sbil_v = 0

    if math.isnan(prezzo_sbil_v):
        prezzo_sbil_v = 0

    if math.isnan(fee_contratto):
        fee_contratto = 0

    return ((delta_sbil_a * (prezzo_sbil_a - prezzo))+(delta_sbil_v * (prezzo + fee_contratto - prezzo_sbil_v)))/(- 100)


def make_tolleranza(delta_tolleranza_acquisto, delta_tolleranza_vendita, prezzo_tolleranza):
    if math.isnan(delta_tolleranza_acquisto):
        delta_tolleranza_acquisto = 0
    if math.isnan(delta_tolleranza_vendita):
        delta_tolleranza_vendita = 0

    a = delta_tolleranza_acquisto * (prezzo_tolleranza + 0.108 * 1.057275)
    b = delta_tolleranza_vendita * (prezzo_tolleranza - 0.108 * 1.057275)
    return (a + b) / 100


def make_costo_gas_totale(costo_gas, sbilancio, tolleranza):
    return costo_gas + sbilancio + tolleranza


def make_costo_gas(miglior_gas, variabili_gas, pcs, prezzo_smc, fee_contratto):
    a = (miglior_gas * variabili_gas) / 100
    b = round((miglior_gas * pcs) / 10.57275)
    c = (prezzo_smc + fee_contratto) / 100
    return a + b * c


def make_incasso_mgp(x):
    a = x['Programma MGP'] * x['MGPPrezzi_NORD']
    b = x['Programma MI1'] * x['MI-A1Prezzi_NORD']
    c = x['Programma MI2'] * x['MI-A2Prezzi_NORD']
    d = x['Programma MI3'] * x['MI-A3Prezzi_NORD']
    if math.isnan(a):
        a = 0
    if math.isnan(b):
        b = 0
    if math.isnan(c):
        c = 0
    if math.isnan(d):
        d = 0

    return a + b + c + d


def make_costo_sbilancio_power(x):
    if x['Produzione NEO'] == 0:
        return - x['Programma totale'] * x['MGPPrezzi_NORD']
    else:
        return (x['Produzione NEO'] - x['Programma totale']) * x['MGPPrezzi_NORD']


def make_sbilancio_power(x):
    return x['Produzione NEO'] - x['Programma totale']


def make_ccc_annuale_base(x):
    return gas_sett.annuale_base_mw * (x['MGPPrezzi_PUN'] - x['MGPPrezzi_NORD'] - gas_sett.annuale_base_euro_mwh)


def make_ccc_mese_picco(x):
    if x['Ore di picco']:
        mese = x['Date'].month - 1
        mw = gas_sett.mensile_picco_mw[mese]
        euro_mw = gas_sett.mensile_picco_mw_euro_mwh[mese]
        return mw * (x['MGPPrezzi_PUN'] - x['MGPPrezzi_NORD'] - euro_mw)
    else:
        return 0


def make_ccc_mese_base(x):
    mese = x['Date'].month - 1
    mw = gas_sett.mensile_base_mw[mese]
    euro_mw = gas_sett.mensile_base_mw_euro_mwh[mese]

    return mw * (x['MGPPrezzi_PUN'] - x['MGPPrezzi_NORD'] - euro_mw)


def make_dict_p_and_l_ccc(df):
    dict_p_and_l_giorno_gas = dict()
    to_add = 0
    for index, row in df.iterrows():
        if row['Date'].hour == 5:
            to_add += row['Euro CCC annuale base'] + row['Euro CCC mensile base'] + row['Euro CCC mensile picco']
            dict_p_and_l_giorno_gas[
                dt.datetime(row['Date'].year, row['Date'].month, row['Date'].day) - dt.timedelta(days=1)] = to_add
            to_add = 0
        else:
            to_add += row['Euro CCC annuale base'] + row['Euro CCC mensile base'] + row['Euro CCC mensile picco']

    return dict_p_and_l_giorno_gas


def prezzo_medio_mwh_riepilogo(prezzo_medio):
    return prezzo_medio / 1.057275


def prezzo_medio_riepilogo(costo_gas, miglior_volume):
    if math.isnan(miglior_volume) or miglior_volume == 0:
        return 0
    else:
        return (costo_gas / miglior_volume) * 100


def costo_gas_riepilogo(df):
    return df['Costo gas totale']


def volume_sbilanciato_riepilogo(miglior_gas, nomina):
    return nomina - miglior_gas


def volume_caldaia_riepilogo(mc, smc_verbali):
    if math.isnan(smc_verbali):
        return mc
    else:
        return smc_verbali


def miglior_energia_gas_riepilogo(miglior_gas, pcs):
    return (miglior_gas * pcs) / 1000


def miglior_gas_riepilogo(df):
    return df['Miglior gas']


def volume_nomina_riepilogo(df):
    return df['m3']


def make_giorno_gas_column(data):
    return (data - dt.timedelta(hours=6)).date()


def tonnellate_co2_riepilogo(miglior_gas, volume_caldaia):
    return (miglior_gas - volume_caldaia) * 0.00198


def make_costo_co2_riepilogo(tonnellate, prezzo):
    return tonnellate*prezzo


def make_p_and_l_riepilogo(incasso, costo_co2, costo_sbilancio, costo_gas):
    return incasso - costo_co2 + costo_sbilancio - costo_gas


def make_p_an_l_epurato(caldaia, prezzo_medio, p_and_l):
    return p_and_l + caldaia * prezzo_medio / 100


def prepare_book():
    df_input = input_loader()

    df_input['date'] = df_input.apply(
        lambda x: format_date(x['date']), axis=1)

    df_input['PCS'] = df_input.apply(
        lambda x: make_pcs(x['PCSx'], df_input), axis=1)
    df_input['m3'] = df_input.apply(
        lambda x: make_m3(x['KWh/d']), axis=1)

    df_input.drop(axis=1, inplace=True, columns=['PCSx', 'KWh/d'])

    df_input_variabili = input_variabili_gas_loader()
    df_input = make_fee_contratto_smc_cent_book(df_input, df_input_variabili)
    df_input = make_variabili_gas_book(df_input, df_input_variabili)
    df_spot_gas = get_prezzi_sbilanciamento()

    x = get_prezzi_heren()

    df_input.drop(axis=1, inplace=True,
                  columns=['PSV DA Mid', 'PSV WEEKEND Mid', 'PSV WEEKEND Mid.1', 'PSV DA Mid.1', 'PSV WEEKEND tmp',
                           'Lavorativo/festivo', 'N° Lavorativo', 'PSV WEEKEND', 'Prezzo Heren'])

    df_input['Prezzo Heren'] = df_input.apply(lambda a: make_heren_price(a['date'], x), axis=1)
    df_input['Miglior gas'] = df_input.apply(
        lambda x: make_best_gas_column(x['Verbale Snam'],
                                       x['Allocato'],
                                       x['m3']), axis=1)
    df_input['Volume 38100'] = df_input.apply(
        lambda x: make_volume_38100_column(x['PCS'],
                                           x['Miglior gas']), axis=1)
    df_input['Tolleranza'] = df_input.apply(
        lambda x: make_tollerance_columns(x['date']), axis=1)
    df_input['Delta sbilanciamento vendita'] = df_input.apply(
        lambda x: make_delta_sbilanciamento_vendita(x['Volume 38100'],
                                                    x['m3'],
                                                    x['Tolleranza']), axis=1)
    df_input['Delta sbilanciamento acquisto'] = df_input.apply(
        lambda x: make_delta_sbilanciamento_acquisto(x['Volume 38100'],
                                                     x['m3'],
                                                     x['Tolleranza']), axis=1)
    df_input['Delta tolleranza vendita'] = df_input.apply(
        lambda x: make_delta_tolleranza_vendita(x['PCS'],
                                                x['Miglior gas'],
                                                x['m3'],
                                                x['Tolleranza']), axis=1)
    df_input['Delta tolleranza acquisto'] = df_input.apply(
        lambda x: make_delta_tolleranza_acquisto(x['PCS'],
                                                 x['Miglior gas'],
                                                 x['m3'],
                                                 x['Tolleranza']), axis=1)
    df_input['Prezzo smc 38100'] = df_input.apply(
        lambda x: make_prezzo_smc_38100(x['Prezzo Heren']), axis=1)
    df_input['Prezzo tolleranza'] = df_input.apply(
        lambda x: make_prezzo_tolleranza(x['Prezzo smc 38100'], x['date']), axis=1)
    df_input['Prezzo Sbil acquisto smc cent'] = df_spot_gas.apply(
        lambda x: make_prezzo_sbilanciamento_smc_cent(x['Prezzo Sbil acquisto']), axis=1)
    df_input['Prezzo Sbil vendita smc cent'] = df_spot_gas.apply(
        lambda x: make_prezzo_sbilanciamento_smc_cent(x['Prezzo Sbil vendita']), axis=1)
    df_input['Tolleranza euro'] = df_input.apply(
        lambda x: make_tolleranza(x['Delta tolleranza acquisto'],
                                  x['Delta tolleranza vendita'],
                                  x['Prezzo tolleranza']), axis=1)
    df_input['Sbilancio'] = df_input.apply(
        lambda x: make_sbilancio(x['Delta sbilanciamento acquisto'],
                                 x['Prezzo Sbil acquisto smc cent'],
                                 x['Prezzo smc 38100'],
                                 x['Delta sbilanciamento vendita'],
                                 x['Prezzo Sbil vendita smc cent'],
                                 x['Fee c€/smc 38,1']
                                 ), axis=1)
    df_input['Tolleranza euro'] = df_input.apply(
        lambda x: make_tolleranza(x['Delta tolleranza acquisto'],
                                  x['Delta tolleranza vendita'],
                                  x['Prezzo tolleranza']), axis=1)
    df_input['Costo gas'] = df_input.apply(
        lambda x: make_costo_gas(x['Miglior gas'],
                                 x['Variabili gas'],
                                 x['PCS'],
                                 x['Prezzo smc 38100'],
                                 x['Fee c€/smc 38,1']
                                 ), axis=1)
    df_input['Costo gas totale'] = df_input.apply(
        lambda x: make_costo_gas_totale(x['Costo gas'],
                                        x['Sbilancio'],
                                        x['Tolleranza euro'],
                                        ), axis=1)

    prezzi_spot_power = kta.get_artesian_data_actual([100206897, 100206816, 100207005, 100001425, 100001429, 100209200],
                                              '2021-12-01', '2023-01-01')
    prezzi_spot_power = kta.format_artesian_data(prezzi_spot_power)
    programma = load_programma()
    df_energia_spot = pd.merge(prezzi_spot_power, programma, left_index=True, right_index=True)
    df_energia_spot['Ore di picco'] = df_energia_spot.apply(lambda x: is_peak_time(x['Date']), axis=1)
    df_energia_spot['Incasso MGP'] = df_energia_spot.apply(lambda x: make_incasso_mgp(x), axis=1)
    df_energia_spot['Programma totale'] = df_energia_spot[['Programma MGP',
                                                           'Programma MI1',
                                                           'Programma MI2',
                                                           'Programma MI3']].sum(axis=1)

    df_energia_spot['Costo sbilancio'] = df_energia_spot.apply(lambda x: make_costo_sbilancio_power(x), axis=1)
    df_energia_spot['Sbilancio'] = df_energia_spot.apply(lambda x: make_sbilancio_power(x), axis=1)
    df_energia_spot['Euro CCC annuale base'] = df_energia_spot.apply(lambda x: make_ccc_annuale_base(x), axis=1)
    df_energia_spot['Euro CCC mensile base'] = df_energia_spot.apply(lambda x: make_ccc_mese_base(x), axis=1)
    df_energia_spot['Euro CCC mensile picco'] = df_energia_spot.apply(lambda x: make_ccc_mese_picco(x), axis=1)
    df_energia_spot['Giorno gas'] = df_energia_spot.apply(lambda x: make_giorno_gas_column(x['Date']), axis=1)

    riepilogo = pd.DataFrame()
    riepilogo['Data'] = df_input['date']
    riepilogo['Volume nomina'] = volume_nomina_riepilogo(df_input)
    riepilogo['Miglior volume gas'] = miglior_gas_riepilogo(df_input)
    riepilogo['Miglior energia gas'] = df_input.apply(
        lambda x: miglior_energia_gas_riepilogo(x['Miglior gas'], x['PCS']), axis=1)
    caldaia = input_caldaia()
    riepilogo['Volume caldaia'] = caldaia.apply(lambda x: volume_caldaia_riepilogo(x['mc'], x['smc verbali']), axis=1)
    riepilogo['Volume sbilanciato'] = riepilogo.apply(
        lambda x: volume_sbilanciato_riepilogo(x['Miglior volume gas'], x['Volume nomina']), axis=1)
    riepilogo['Costo gas'] = costo_gas_riepilogo(df_input)
    riepilogo['Prezzo medio'] = riepilogo.apply(
        lambda x: prezzo_medio_riepilogo(x['Costo gas'], x['Miglior volume gas']), axis=1)
    riepilogo['Prezzo medio MWh'] = riepilogo.apply(lambda x: prezzo_medio_mwh_riepilogo(x['Prezzo medio']), axis=1)
    df_energia_spot['Giorno gas'] = df_energia_spot.apply(lambda x: make_giorno_gas_column(x['Date']), axis=1)
    z = df_energia_spot.groupby('Giorno gas').sum()
    y = riepilogo.set_index('Data').join(z)
    y.drop([pd.NaT], inplace=True)
    riepilogo.set_index('Data', drop=False, inplace=True)
    riepilogo['Energia programmata'] = y['Programma totale']
    riepilogo['Energia prodotta'] = y['Produzione NEO']
    riepilogo['Costo sbilancio'] = y['Costo sbilancio']
    riepilogo['Incasso mercato'] = y['Incasso MGP']
    riepilogo['Tonnellate Co2'] = riepilogo.apply(
        lambda x: tonnellate_co2_riepilogo(x['Miglior volume gas'], x['Volume caldaia']), axis=1)
    riepilogo.drop([pd.NaT], inplace=True)
    data_fine_estrazione = riepilogo.iloc[-1]['Data'].date()
    data_fine_estrazione = data_fine_estrazione + dt.timedelta(days=1)
    data_fine_estrazione = data_fine_estrazione.strftime('%Y-%m-%d')
    data_inizio_estrazione = riepilogo.iloc[0]['Data'].date().strftime('%Y-%m-%d')
    co2 = kta.format_artesian_data(kta.get_artesian_data_actual_daily([100209203],
                                                               data_inizio_estrazione, data_fine_estrazione))
    riepilogo['Prezzo Co2'] = co2['CO2 KtE'].values
    riepilogo['Costo Co2'] = riepilogo.apply(
        lambda x: make_costo_co2_riepilogo(x['Tonnellate Co2'], x['Prezzo Co2']), axis=1)
    riepilogo['CCC'] = y['Euro CCC mensile base'] + y['Euro CCC mensile picco'] + y['Euro CCC annuale base']
    riepilogo['P&L'] = riepilogo.apply(
        lambda x: make_p_and_l_riepilogo(x['Incasso mercato'], x['Costo Co2'], x['Costo sbilancio'], x['Costo gas']),
        axis=1)
    riepilogo['P&L epurato'] = riepilogo.apply(
        lambda x: make_p_an_l_epurato(x['Volume caldaia'], x['Prezzo medio'], x['P&L']),
        axis=1)

    return df_input, df_energia_spot, riepilogo, y


def readCSVFromAzureBlobStorage(strStorageAccountURL, strStorageAccountKey, strContainerName, strBlobName, strFilename, index_col=None, sheet_name="input generici", header=0, usecols=None, skiprows=None):
    blob_service_client_instance = BlobServiceClient(account_url = strStorageAccountURL, credential = strStorageAccountKey)
    blob_client_instance = blob_service_client_instance.get_blob_client(strContainerName, strBlobName + strFilename, snapshot = None)
    with open(strFilename, "wb") as my_blob:
        blob_data = blob_client_instance.download_blob()
        blob_data.readinto(my_blob)
        pdOut = pd.read_excel(strFilename, index_col=index_col, sheet_name=sheet_name, header=header,usecols=usecols, skiprows=skiprows)
        os.remove(strFilename)
    return pdOut

