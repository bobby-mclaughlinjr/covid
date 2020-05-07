import pandas as pd
import numpy as np
import datetime
from functools import wraps

SOURCES = ['CSSE', 'NYT']

CSSE_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
CSSE_COLUMNS = {'Province/State': 'area', 'Country/Region': 'region', 'Lat': 'latitude', 'Long': 'longitude'}

NYT_URL_PREFIX = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-'

FILTERED_REGIONS = ['US', 'Spain', 'Italy', 'France', 'Germany', 'United Kingdom', 'Sweden', 'South Korea', 'Japan', 'Singapore'
                   , 'Denmark', 'Australia', 'California - US', 'Florida - US', 'Georgia - US', 'Illinois - US', 'Louisiana - US'
                   , 'Massachusetts - US', 'Michigan - US', 'New York - US', 'New Jersey - US', 'Pennsylvania - US', 'Texas - US']


def smooth_diff(X, window=None, diff=None):
    if window is not None:
        X = X.rolling(window=window).mean()[window:]

    if diff is not None:
        X = X.diff(diff)[diff:]

    return X


def reset_drop(func):
    @wraps(func)
    def wrapper(X, **kwargs):
        X.reset_index(level=0, drop=True, inplace=True)
        return func(X, **kwargs)

    return wrapper


class Covid(object):

    """Retrieving & manipulating data (cases, deaths, test, etc) related to Covid-19"""

    def __init__(self, regions=None):
        self.regions = regions
        self.data = None

    @property
    def regions(self):
        return self._regions

    @regions.setter
    def regions(self, value):
        if value is None: value = FILTERED_REGIONS
        if not isinstance(value, list): value = [value]

        self._regions = value

    @property
    def single(self):
        return len(self.regions) == 1

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if self.single & (value is not None):
            value.reset_index(level=0, drop=True, inplace=True)

        self._data = value

    @staticmethod
    def get_CSSE():
        df = pd.read_csv(CSSE_URL).rename(columns=CSSE_COLUMNS)
        data = pd.melt(df, id_vars=CSSE_COLUMNS.values(), var_name='date', value_name='cases')
        data['date'] = [datetime.datetime.strptime(str(date), '%m/%d/%y') for date in data['date']]

        return data.groupby(['region', 'date'])['cases'].sum()

    @staticmethod
    def get_NYT():
        data = pd.read_csv(NYT_URL_PREFIX + 'states.csv').rename(columns={'state': 'region'})
        data['region'] = [region + ' - US' for region in data['region']]
        data['date'] = [datetime.datetime.strptime(str(date), '%Y-%m-%d') for date in data['date']]

        return data.groupby(['region', 'date'])['cases'].sum()

    @staticmethod
    def get_NYC():
        data = pd.read_csv(NYT_URL_PREFIX + 'counties.csv')
        data = data[data['county'] == 'New York City']
        data['region'] = data['county'] + ' - ' + data['state']

        data['date'] = [datetime.datetime.strptime(str(date), '%Y-%m-%d') for date in data['date']]

        return data.groupby(['region', 'date'])['cases'].sum()

    @staticmethod
    @reset_drop
    def smooth_diff(X, window=None, diff=None):
        return smooth_diff(X, window=window, diff=diff)

    @staticmethod
    @reset_drop
    def outbreak_shift(X, n=15):
        min_index = X.sort_index().index[0]
        add_series = pd.Series(np.zeros(n), index=[min_index - datetime.timedelta(x) for x in range(1, n+1)])

        return X.append(add_series).sort_index()

    def get_data(self, outbreak_shift=None, smooth=None, diff=None):
        data = pd.concat([self.get_CSSE(), self.get_NYT(), self.get_NYC()]).loc[self.regions, :].sort_index()

        if outbreak_shift is not None:
            data = data.groupby(level=0).apply(self.outbreak_shift, n=outbreak_shift)

        data = data.groupby(level=0).apply(self.smooth_diff, window=smooth, diff=diff)

        self.data = data
        return self
