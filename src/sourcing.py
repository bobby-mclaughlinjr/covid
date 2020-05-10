import pandas as pd
import numpy as np
import datetime
from functools import wraps

import requests
from bs4 import BeautifulSoup
from us.states import lookup as us_state_lookup

# Cases
CASES_SOURCES = ['CSSE', 'NYT']

CSSE_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
CSSE_COLUMNS = {'Province/State': 'area', 'Country/Region': 'region', 'Lat': 'latitude', 'Long': 'longitude'}

NYT_URL_PREFIX = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-'

# Testing
TESTING_SOURCES = ['COVID_TRACKING', 'OUR_WORLD']

COVID_TRACKING_URL = 'https://covidtracking.com/api/v1/states/daily.json'

OUR_WORLD_IN_DATA_URL = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/testing/covid-testing-all-observations.csv'
OUR_WORLD_IN_DATA_COLUMNS = {'Date': 'date', 'Entity': 'region', 'Cumulative total': 'tests', 'Daily change in cumulative total': 'new'
                             , 'Cumulative total per thousand': 'tests_per_thousand', 'Daily change in cumulative total per thousand': 'new_per_thousand'}

FILTERED_REGIONS = ['US', 'Spain', 'Italy', 'France', 'Germany', 'United Kingdom', 'Sweden', 'South Korea', 'Japan', 'Singapore'
                   , 'Denmark', 'Australia', 'California - US', 'Florida - US', 'Georgia - US', 'Illinois - US', 'Louisiana - US'
                   , 'Massachusetts - US', 'Michigan - US', 'New York - US', 'New Jersey - US', 'Pennsylvania - US', 'Texas - US']

US_STATE_SUFFIX = ' - US'


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


def reindex_freq(x):
    redux = x.groupby('date').mean().reindex(pd.date_range(x.index.min(), x.index.max(), freq='D'))
    redux.index.name = 'date'
    return redux


def us_state_populations():
    soup = BeautifulSoup(requests.get('https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States_by_population').text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})

    populations = {}
    for tr in table.find_all('tr'):
        tds = tr.find_all('td')
        if not tds:
            continue

        current_rank, previous_rank, state, population = [td.text.strip() for td in tds[:4]]
        populations[state] = float(population.replace(',', ''))

    return populations


class Covid(object):

    """Retrieving & manipulating data (cases, deaths, test, etc) related to Covid-19"""

    def __init__(self, regions=None):
        self.regions = regions
        self.cases = None
        self.tests = None

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
    def cases(self):
        return self._cases

    @cases.setter
    def cases(self, value):
        if self.single & (value is not None):
            value.reset_index(level=0, drop=True, inplace=True)

        self._cases = value

    @staticmethod
    def get_csse():
        df = pd.read_csv(CSSE_URL).rename(columns=CSSE_COLUMNS)
        data = pd.melt(df, id_vars=CSSE_COLUMNS.values(), var_name='date', value_name='cases')
        data['date'] = [datetime.datetime.strptime(str(date), '%m/%d/%y') for date in data['date']]

        return data.groupby(['region', 'date'])['cases'].sum()

    @staticmethod
    def get_nyt():
        data = pd.read_csv(NYT_URL_PREFIX + 'states.csv').rename(columns={'state': 'region'})
        data['region'] = [region + US_STATE_SUFFIX for region in data['region']]
        data['date'] = [datetime.datetime.strptime(str(date), '%Y-%m-%d') for date in data['date']]

        return data.groupby(['region', 'date'])['cases'].sum()

    @staticmethod
    def get_nyc():
        data = pd.read_csv(NYT_URL_PREFIX + 'counties.csv')
        data = data[data['county'] == 'New York City']
        data['region'] = data['county'] + ' - ' + data['state']

        data['date'] = [datetime.datetime.strptime(str(date), '%Y-%m-%d') for date in data['date']]

        return data.groupby(['region', 'date'])['cases'].sum()

    @staticmethod
    def get_ourworld(per_population=False):
        data = pd.read_csv(OUR_WORLD_IN_DATA_URL, parse_dates=['Date']).rename(columns=OUR_WORLD_IN_DATA_COLUMNS)[OUR_WORLD_IN_DATA_COLUMNS.values()]
        data[['region', 'units']] = data['region'].apply(lambda x: pd.Series(str(x).split(' - ')))

        if per_population:
            data = data[['region', 'date', 'tests_per_thousand']].rename(columns={'tests_per_thousand': 'tests'})

        return data.set_index('date').groupby('region')['tests'].apply(reindex_freq).groupby(['region', 'date']).sum()

    @staticmethod
    def get_covid_tracking(per_population=False):
        data = pd.read_json(COVID_TRACKING_URL).rename(columns={'total': 'tests'})
        data['date'] = [datetime.datetime.strptime(str(date), '%Y%m%d') for date in data['date']]
        data['state'] = [str(us_state_lookup(state)) for state in data['state']]

        if per_population:
            populations = us_state_populations()
            data['population'] = [populations.get(state, None) for state in data['state']]
            data['tests'] = data['tests'] / (data['population'] / 1000)

        data['state'] = [state + US_STATE_SUFFIX for state in data['state']]

        return data.rename(columns={'state': 'region'}).groupby(['region', 'date'])['tests'].sum()

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

    def get_cases(self, outbreak_shift=None, smooth=None, diff=None):
        data = pd.concat([self.get_csse(), self.get_nyt(), self.get_nyc()]).loc[self.regions, :].sort_index()

        if outbreak_shift is not None:
            data = data.groupby(level=0).apply(self.outbreak_shift, n=outbreak_shift)

        data = data.groupby(level=0).apply(self.smooth_diff, window=smooth, diff=diff)

        self.cases = data
        return self

    def get_tests(self, per_population=False):
        self.tests = pd.concat([self.get_ourworld(per_population), self.get_covid_tracking(per_population)]).loc[self.regions, :].sort_index()
        return self
