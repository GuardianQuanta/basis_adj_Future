import time
from genericpath import isfile
from GQDatabaseSQL.MDSDB import MDSDB
from GQDatabaseSQL import SETSMART
import numpy as np
import pandas as pd
import TFEX_Utils, os
from multiprocessing import Pool
from utils import utils
# write_path ="C:\\gitrepos\\DataWriter\\DataCache\\S50"
# write_path ="L:\\aj_future_datacache\\S50\\Offset\\4T"
write_path ="X:\\S50\\FUT\\raw"
write_path_roll ="X:\\S50\\FUT\\raw_roll"
# write_path ="K:\\S50FUT\\raw\\"
Freq = "5T"
# Freq = "30S"

write_path = write_path +"\\" +Freq
write_path_roll = write_path_roll + "\\" +Freq
offset = None
get_weighted_roll = True

def dowork(params):

    date_pair_array, symbols_prefix, Freq, holidays = params
    for date_pair in date_pair_array:
        # print(f"i {i} {indices[i]}")
        # print(f"i-1 {i-1} {indices[i-1]}")
        starting_period_dt = date_pair[0]
        ending_period_dt = date_pair[1]
        print(f"starting: {starting_period_dt}")
        print(f"ending_period: {ending_period_dt}")
        # Freq = "15T"
        # Rolling_D = 15
        print(f"NDays: {(ending_period_dt - starting_period_dt).days }")

        filedate = (
            starting_period_dt.strftime("%Y-%m-%d")
            + "_"
            + ending_period_dt.strftime("%Y-%m-%d")
        )


        if os.path.isfile(os.path.join(write_path, filedate + f"_FUT_{Freq}.fea")):
            print(f"Skipping: {filedate}")
            continue
        else:
            print(f"running {filedate}")

        '''get_tfex_future_fixed_int_OHLCV_LOB_with_offset is the correct function'''
        S50F_LOB_DF = MDSDB.get_tfex_future_fixed_int_OHLCV_LOB(
            date_1=starting_period_dt,
            date_2=ending_period_dt,
            symbol=symbols_prefix,
            block_time=Freq,
            max_days=15
        )

        Rolling_D = 15
        # Get Weighted Rolling Future
        if get_weighted_roll:
            FinalWeightFuture = TFEX_Utils.TFEX_Utils.S50_tfex_weighted_rolling_future_OHLCV_QuartersOnly(
                S50F_LOB_DF, holidays=holidays,N_roll_days=Rolling_D
            )

        contract_specifications = SETSMART.get_TFEX_series_INFO_by_underlying_info(
            _underlying_prefix=symbols_prefix, _instrument_type=-1
        )

        contract_specifications.index = [
            name.rstrip() for name in contract_specifications["N_SERIES"]
        ]
        Expirations_header = contract_specifications.loc[
            S50F_LOB_DF.columns.get_level_values(0), "D_EXPIRED"
        ]

        header_with_exp = pd.MultiIndex.from_arrays(
            [
                Expirations_header,
                S50F_LOB_DF.columns.get_level_values(0),
                S50F_LOB_DF.columns.get_level_values(1),
                S50F_LOB_DF.columns.get_level_values(2),
            ]
        )

        S50F_LOB_DF.columns = [
            col[0].strftime("%Y-%m-%d") + "_" + col[1] + "_" + col[2] + "_" + col[3]
            for col in header_with_exp
        ]
        # S50F_LOB_DF.reset_index().to_feather(
        #     os.path.join(write_path, filedate + f"_FUT_{Freq}.fea")
        # )
        # S50_Option_LOB_DF.to_feather(os.path.join(write_path,filedate + f"_OPT_{Freq}.fea"))

        pt1 = time.time()
        if offset is None:
            if get_weighted_roll:
                FinalWeightFuture.reset_index().to_feather(
                    os.path.join(write_path, filedate + f"_FUTROLL_{Freq}.fea")
                )
            S50F_LOB_DF.reset_index().to_feather(
                os.path.join(write_path, filedate + f"_FUT_{Freq}.fea")
            )
        else:
            if get_weighted_roll:
                FinalWeightFuture.reset_index().to_feather(
                    os.path.join(write_path, filedate + f"_FUTROLL_{Freq}_{offset}.fea")
                )
            S50F_LOB_DF.reset_index().to_feather(
                os.path.join(write_path, filedate + f"_FUT_{Freq}_{offset}.fea")
            )
        pt2 = time.time()
        print(f"write time: {pt2-pt1}")

def get_filefullpath(start_dt, end_dt, Freq,offset=None):
    filedate = (
            start_dt.strftime("%Y-%m-%d")
            + "_"
            + end_dt.strftime("%Y-%m-%d")
    )

    if offset:
        full_file_path = os.path.join(write_path,  filedate + f"_FUT_{Freq}_{offset}.fea")

    else:
        full_file_path = os.path.join(write_path,  filedate + f"_FUT_{Freq}.fea")
    return full_file_path


if __name__ == "__main__":

    start_dt = pd.to_datetime("20190904")

    # start_dt = pd.to_datetime("20230327")
    # end_date = pd.to_datetime("20231010")
    # end_date = pd.to_datetime("20230909")
    # start_dt = pd.to_datetime("20231019")
    end_date = pd.to_datetime(pd.Timestamp.now().date())

    get_weekly_dates_for_data = utils.get_weekly_dates_for_data(start_dt,end_date)

    holiday_class = TFEX_Utils.TFEX_Utils.SETSMART_Holidays()
    holidays = holiday_class.get_holidays()

    symbol_pref ="S50___"
    SkipExisting = False
    date_pair_array = []
    for i in range(1,len(get_weekly_dates_for_data)):

        full_file_path = get_filefullpath(get_weekly_dates_for_data[i-1], get_weekly_dates_for_data[i],
                                          Freq=Freq,offset=offset)
        if SkipExisting and os.path.isfile(full_file_path):
            print(f"Skipping {full_file_path}")
            continue

        date_pair_array.append([get_weekly_dates_for_data[i-1],get_weekly_dates_for_data[i]])

    # Freq_int = 15
    n_jobs = 4
    ArrayChunks = np.array_split(date_pair_array,n_jobs)


    # dowork( (date_pair_array,symbol_pref,Freq,holidays))
    # ArrayChunks[0]
    p = Pool(processes=n_jobs)
    OutputArray = p.map(dowork, zip( ArrayChunks, [symbol_pref]*n_jobs, [Freq] *n_jobs,[holidays]*n_jobs))
    # dowork((ArrayChunks[0],"S50%",Freq,holidays))

    p.close()

