import KTE_Gas as ktg
import KTE_artesian as kta
import datetime as dt
import pandas as pd

path = ''

book, energia_spot, riepilogo, y = ktg.prepare_book()

list_column_names_to_upload = list(riepilogo.columns)
list_column_names_to_upload.remove('Volume nomina')
list_column_names_to_upload.remove('Data')
for column in list_column_names_to_upload:
    dict_to_upload = kta.get_artesian_dict_versioned_daily_by_index(riepilogo.fillna(0), column)
    kta.post_artesian_versioned_time_series(dict_to_upload, dict(), 'DevKtE', column + ' book UP_CETSERVOLA_1', 'd')

with pd.ExcelWriter(path + dt.datetime.today().strftime('%Y-%m-d') + ' Book UP_CETSERVOLA_1.xlsx') as writer:
    book.to_excel(writer, sheet_name='Book')
    riepilogo.to_excel(writer, sheet_name='Riepilogo')

energia_spot.to_csv(path + 'Energia spot.csv ')
