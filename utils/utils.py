import pandas as pd
from TFEX_Utils import TFEX_Utils
def get_weekly_dates_for_data(start_date,end_date):
    date_range_index = pd.date_range(start_date, end_date, freq='W-FRI')
    date_range_index = pd.DatetimeIndex([start_date]).append(date_range_index.append(pd.DatetimeIndex([end_date])))

    return date_range_index


if __name__ == "__main__":
    holiday_class = TFEX_Utils.SETSMART_Holidays()
    holidays = holiday_class.get_holidays()

    print()
