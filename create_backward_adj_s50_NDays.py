import os, sys,calendar
import pandas as pd
import numpy as np
import gc
import datetime
import TFEX_Utils
from utils.utils import (S50_tfex_roll_basis_OHLCV_QuartersOnly)
import configparser
import matplotlib.pyplot as plt
# from GQDatabaseSQL import

appconfig = configparser.ConfigParser()
appconfig.read("..\\AppConfig.ini")

def get_transform_columns(columns,Holiday):
    column_names = np.array([colname.split("_") for colname in columns])

    expiration_dates = [ TFEX_Utils.TFEX_Utils.GetExpFromSym(series_sym,pd.to_datetime(Holiday.values)) for series_sym  in column_names[:, 0]]
    expiration_dates = pd.to_datetime(expiration_dates)
    return pd.MultiIndex.from_arrays(
                [
                    expiration_dates,
                    column_names[:, 0],
                    column_names[:, 1],
                    column_names[:, 2],
                ]
            )


raw_path = "X:\\S50\\FUT\\raw\\5T"
if __name__ =="__main__":
    Holiday_class = TFEX_Utils.TFEX_Utils.SETSMART_Holidays()
    Holidays = Holiday_class.get_holidays()

    df_array = []
    for file in os.listdir(raw_path):
        # print()

        df = pd.read_feather(os.path.join(raw_path, file))
        df.set_index('ExTime',inplace=True)
        df.columns = get_transform_columns(df.columns,Holidays)
        df = TFEX_Utils.TFEX_Utils.remove_holidays_weekend_closinghour(df, holidays=pd.to_datetime(
            Holidays.values.reshape(-1)), UTC=True)

        df_array.append(df)
    DF = pd.concat(df_array)
    DF_unadjusted,basis_diff_Array, basis_diff_Array_cumsum_reverse, = S50_tfex_roll_basis_OHLCV_QuartersOnly(DF, pd.to_datetime(Holidays.values.reshape(-1)),roll_days=3)

    DF_unadjusted.loc[:,['LastPrice_open','LastPrice_high','LastPrice_low','LastPrice_close','Volume_sum']].to_csv("BasisAdjFuture_check_OHLCV.csv")
    basis_diff_Array_cumsum_reverse.to_csv("reversed_basis_cumsum.csv")
    print(DF.tail())


