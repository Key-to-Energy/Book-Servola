import logging
import KTE_Gas as ktg
import KTE_artesian as kta
import datetime as dt
import pandas as pd
import azure.functions as func


def main(myblob: func.InputStream):
    logging.info('Script partito')
    book, energia_spot, riepilogo, y = ktg.prepare_book()
    logging.info('Book generato')
    list_column_names_to_upload = list(riepilogo.columns)
    list_column_names_to_upload.remove('Volume nomina')
    list_column_names_to_upload.remove('Data')
    logging.info('Caricando i dati su Artesian')
    for column in list_column_names_to_upload:
        dict_to_upload = kta.get_artesian_dict_versioned_daily_by_index(riepilogo.fillna(0), column)
        kta.post_artesian_versioned_time_series(dict_to_upload, dict(), 'DevKtE', column + ' book UP_CETSERVOLA_1', 'd')


    logging.info(f"Python blob trigger function processed blob \n"
                 f"Name: {myblob.name}\n"
                 f"Blob Size: {myblob.length} bytes")
