import datetime
import time

import TFEX_Utils.TFEX_Utils
import numpy as np
import pandas as pd
import os,sys
import threading
import logging

'''
Rule#1
At least Time to expiration is less than 5 days
look for the date when seoond quarter expiration trading volume > front expiration trading volume
'''

def S50_tfex_roll_basis_OHLCV_QuartersOnly(
    LOB_data_resampled,holidays, roll_days = 1):
    idx = pd.IndexSlice
    Quarter_only_index = LOB_data_resampled.columns.get_level_values(0).month % 3 == 0
    QData = LOB_data_resampled.iloc[:, np.where(Quarter_only_index)[0]]
    QData = QData.sort_index(axis=1, level=0)
    # QData.columns.get_level_values(3)

    Expirations_in_data = QData.columns.get_level_values(0).unique()

    ''''''
    tile_exp_date = np.tile(pd.to_datetime(Expirations_in_data.date), QData.shape[0]).reshape(
        QData.shape[0], -1
    )

    DaysToExp = [np.busday_count(list(QData.index.date), list(pd.to_datetime(tile_exp_date[:, i_i]).date),
                                 holidays=list(holidays.date)).reshape(-1, 1) for i_i in
                 range(Expirations_in_data.shape[0])]
    DaysToExp = np.hstack(DaysToExp).astype(float)

    # close_data =
    close_data_diff = QData.loc[:,idx[:,:,'LastPrice',"close"]].diff(periods=-1, axis=1)
    ''''''
    # pd.DataFrame(DaysToExp, index=QData.index, columns=pd.to_datetime(Expirations_in_data.date)).to_csv("DaysToExp.csv")

    DaysToExp[DaysToExp < roll_days] = np.nan
    front_exp_indices = np.nanargmin(DaysToExp, axis=1)
    # second_exp_indices = front_exp_indices + 1
    roll_index = np.append([0],np.diff(front_exp_indices))

    roll_index_where = np.where(roll_index == 1)[0]-1

    basis_diff = close_data_diff.values[roll_index_where,front_exp_indices[roll_index_where]]
    basis_diff_Array = np.zeros(QData.shape[0])
    basis_diff_Array[roll_index_where] = basis_diff * -1
    basis_diff_Array = np.cumsum(basis_diff_Array[::-1])[::-1]

    weight_matrix = np.zeros(DaysToExp.shape)
    weight_matrix[(np.arange(DaysToExp.shape[0]), front_exp_indices)] = 1

    weight_matrix = pd.DataFrame(
        weight_matrix, index=QData.index, columns=Expirations_in_data
    )
    weight_matrix = weight_matrix.reindex(QData.columns.get_level_values(0), axis=1)
    weight_qData = QData * weight_matrix.values

    # flatten_columns = (
    #     weight_qData.columns.get_level_values(2)
    #     + "_"
    #     + weight_qData.columns.get_level_values(3)
    # )

    columns_to_adj = ['LastPrice',
                      'MDAsk1Price','MDAsk2Price','MDAsk3Price','MDAsk4Price','MDAsk5Price',
                      'MDBid1Price','MDBid2Price','MDBid3Price','MDBid4Price','MDBid5Price',
                      "VWAP"
                      ]

    weight_future = weight_qData.groupby(axis=1, level=[2,3]).sum()

    weight_future.loc[:,idx[columns_to_adj,:]] = weight_future.loc[:,idx[columns_to_adj,:]] + basis_diff_Array.reshape(-1,1)


    flatten_columns = (
        weight_future.columns.get_level_values(0)
        + "_"
        + weight_future.columns.get_level_values(1)
    )
    weight_future.columns = flatten_columns
    # weight_future = weight_future + basis_diff_Array.reshape(-1,1)
    # print()
    return weight_future

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

if __name__ == "__main__":
    print()

    Holiday_class = TFEX_Utils.TFEX_Utils.SETSMART_Holidays()
    Holidays = Holiday_class.get_holidays()

    df_array = []
    basis_adj_array = []

    # np.sort(os.listdir(raw_path))
    for file in np.sort(os.listdir(raw_path)):
        # print()
        filesplit = file.split('_')

        df = pd.read_feather(os.path.join(raw_path, file))
        df.set_index('ExTime',inplace=True)
        df.columns = get_transform_columns(df.columns,Holidays)

        DF = TFEX_Utils.TFEX_Utils.remove_holidays_weekend_closinghour(df,holidays=pd.to_datetime(Holidays.values.reshape(-1)),UTC=True)

        min_roll_days = 5

        idx = pd.IndexSlice
        Quarter_only_index = DF.columns.get_level_values(0).month % 3 == 0
        QData = DF.iloc[:, np.where(Quarter_only_index)[0]]
        QData = QData.sort_index(axis=1, level=0)
        # QData.columns.get_level_values(3)

        Expirations_in_data = QData.columns.get_level_values(0).unique()

        ''''''
        tile_exp_date = np.tile(pd.to_datetime(Expirations_in_data.date), QData.shape[0]).reshape(
        QData.shape[0], -1
        )

        DaysToExp = [np.busday_count(list(QData.index.date), list(pd.to_datetime(tile_exp_date[:, i_i]).date),
                                     holidays=list(Holidays.date)).reshape(-1, 1) for i_i in
        range(Expirations_in_data.shape[0])]
        DaysToExp = np.hstack(DaysToExp).astype(float)

        '''Check for rolling conodition'''
        if np.any(DaysToExp[:,0] <= min_roll_days):
            '''Condition 1 Front exp is less than  min_roll_days'''
            Volume_sum = QData.loc[:, idx[Expirations_in_data[:2], :, 'Volume', 'sum']]

            Volume_sum_by_days_to_exp = Volume_sum.groupby(DaysToExp[:, 0]).sum()
            Volume_sum_by_days_to_exp = Volume_sum_by_days_to_exp[Volume_sum_by_days_to_exp.index<=min_roll_days]

            if len(basis_adj_array)> 0:
                '''Checking for recent roll'''
                last_basis = basis_adj_array[-1]

                if np.any(~np.isnan(last_basis)):
                    '''If there is any basis that is not nan, there is a basis adjustment in the last iteration
                    must use second expiration'''
                    second_exp = Expirations_in_data.sort_values()[1]
                    target_df = QData.loc[:, second_exp].droplevel(0, axis=1)

                    target_df.columns = [
                        col[0] + "_" + col[1]
                        for col in target_df.columns
                    ]

                    basis_series = pd.Series(np.zeros(target_df.shape[0]), index=target_df.index)

                    df_array.append(target_df)
                    basis_adj_array.append(basis_series)

                    #go next
                    continue
                else:
                    print('No recent rolling, moving on to normal case')
                    pass

            if (Volume_sum_by_days_to_exp.iloc[:, 1] > Volume_sum_by_days_to_exp.iloc[:,0]).any():
                '''Found date that second exp volume > front exp'''
                roll_days = Volume_sum_by_days_to_exp.index[(Volume_sum_by_days_to_exp.iloc[:, 1] > Volume_sum_by_days_to_exp.iloc[:,0])].max()


                '''Geting Indices for each expirations'''
                second_quarter_index = DaysToExp[:, 0] < roll_days
                front_quarter_index = DaysToExp[:, 0] >= roll_days

                basis_calc_index = np.where(front_quarter_index)[0][-1]

                second_exp = Expirations_in_data.sort_values()[1]
                target_df = QData.loc[:, second_exp].droplevel(0, axis=1)
                target_df = target_df[second_quarter_index]


                # if front_quarter_index[front_quarter_index].shape[0] > 0:
                front_exp = Expirations_in_data.sort_values()[0]
                front_target_df = QData.loc[:, front_exp].droplevel(0, axis=1)
                front_target_df = front_target_df[front_quarter_index]
                target_df = pd.concat([front_target_df,target_df],axis=0)

                '''Calculating Basis changes to use as adjustment'''
                basis = QData.iloc[basis_calc_index].loc[second_exp].droplevel(0, axis=0).loc[idx['LastPrice', 'close']] - \
                    QData.iloc[basis_calc_index].loc[front_exp].droplevel(0, axis=0).loc[idx['LastPrice', 'close']]
                basis = np.round(basis,2) #rounding python floating precision errors e.i. (149.9999999998)

                basis_series = pd.Series(np.zeros(target_df.shape[0])*np.nan,index=target_df.index )
                basis_series[basis_calc_index] = basis

                target_df.columns = [
                    col[0] + "_" + col[1]
                    for col in target_df.columns
                ]


                df_array.append(target_df)
                basis_adj_array.append(basis_series)
            else:
                '''Does not fit our condition for rolling yet'''
                front_exp = Expirations_in_data.sort_values()[0]
                target_df = QData.loc[:, front_exp].droplevel(0, axis=1)

                target_df.columns = [
                    col[0] + "_" + col[1]
                    for col in target_df.columns
                ]
                df_array.append(target_df)

                basis_series = pd.Series(np.zeros(target_df.shape[0])*np.nan,index=target_df.index )
                basis_adj_array.append(basis_series)

        else:
            '''Does not fit our 1st condition for rolling, You'd take the front Quarter only'''
            # print()
            front_exp = Expirations_in_data.sort_values()[0]
            target_df = QData.loc[:,front_exp].droplevel(0,axis=1)

            target_df.columns = [
                col[0] + "_" + col[1]
                for col in target_df.columns
            ]
            df_array.append(target_df)

            basis_series = pd.Series(np.zeros(target_df.shape[0])*np.nan, index=target_df.index)
            basis_adj_array.append(basis_series)


    DF = pd.concat(df_array).sort_index()
    Basis_ADJ_series = pd.concat(basis_adj_array).sort_index()

    reversed_basis_cumsum = Basis_ADJ_series.fillna(0)[::-1].cumsum()[::-1]
    # print(DF.tail())

    Basis_ADJ_series = pd.DataFrame(Basis_ADJ_series,columns=['BasisADJ'])
    reversed_basis_cumsum = pd.DataFrame(reversed_basis_cumsum,columns=['PriceAdjustmentSeries'])

    reversed_basis_cumsum.to_parquet("PriceAdjustmentSeries.par")
    Basis_ADJ_series.to_parquet("Basis_ADJ_series.par")
    DF.to_parquet("BasisAdjFuture.par")
    print()
    DF.loc[:,['LastPrice_open','LastPrice_high','LastPrice_low','LastPrice_close','Volume_sum']].to_csv("BasisAdjFuture_check_OHLCV.csv")
    reversed_basis_cumsum.to_csv("reversed_basis_cumsum.csv")
