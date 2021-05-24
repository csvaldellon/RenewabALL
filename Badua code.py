import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

df = pd.read_csv("D:/DS Contests/1 Accenture/4 Datasets by Others/DOE data 3.csv",
                 index_col=0)
df.index = pd.to_datetime(df.iloc[:, 0])
df = df.iloc[:, 1:]
df = df.sort_values(by='YYYY-MM')
df.index = df.index.to_period("M")
Luzon = df[df['Grid'] == 'Luzon'].iloc[:, 0]
Visayas = df[df['Grid'] == 'Visayas'].iloc[:, 0]
Mindanao = df[df['Grid'] == 'Mindanao'].iloc[:, 0]

from warnings import simplefilter

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import (
    EnsembleForecaster,
    MultiplexForecaster,
    TransformedTargetForecaster,
    make_reduction,
)
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.model_selection import (
    ExpandingWindowSplitter,
    ForecastingGridSearchCV,
    SlidingWindowSplitter,
    temporal_train_test_split,
)
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.performance_metrics.forecasting import (
    MeanAbsolutePercentageError,
    mean_absolute_percentage_error,
)
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from sktime.utils.plotting import plot_series
from sklearn.neighbors import KNeighborsRegressor
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.arima import ARIMA, AutoARIMA
from sktime.forecasting.bats import BATS

simplefilter("ignore", FutureWarning)


def split_data(Series, points_to_predict):
    y_train, y_test = temporal_train_test_split(Series, test_size=points_to_predict)
    fh = np.arange(len(y_test)) + 1
    print(y_train.shape[0], y_test.shape[0])
    return y_train, y_test, fh


def Naive_sp_1(split_data_function):  # uses result of the split_data function
    # Example: Naive_sp_1((split_data(Luzon, 40))
    y_train, y_test, fh = split_data_function
    forecaster = NaiveForecaster(strategy="last")
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)
    plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
    print(mean_absolute_error(y_pred, y_test))


def Naive_sp_12(split_data_function):
    y_train, y_test, fh = split_data_function
    forecaster = NaiveForecaster(strategy="last", sp=12)
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)
    plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
    print(mean_absolute_error(y_pred, y_test))


def KNeighbors_Regressor(split_data_function, nneighbors=1, windowlength=15, Strategy="recursive"):
    y_train, y_test, fh = split_data_function
    regressor = KNeighborsRegressor(n_neighbors=nneighbors)
    forecaster = make_reduction(regressor, window_length=windowlength, strategy=Strategy)
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)
    plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
    print(mean_absolute_error(y_pred, y_test))


def exp_smoothing(split_data_function, Trend='add', Seasonal="additive", SP=12):
    y_train, y_test, fh = split_data_function
    forecaster = ExponentialSmoothing(trend=Trend, seasonal=Seasonal, sp=SP)
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)
    plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
    plt.show()
    print(mean_absolute_error(y_pred, y_test))


def ETS(split_data_function, Auto=True, SP=12, njobs=-1):
    y_train, y_test, fh = split_data_function
    forecaster = AutoETS(auto=Auto, sp=SP, n_jobs=njobs)
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)
    plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
    plt.show()
    print(mean_absolute_error(y_pred, y_test))


def Arima(split_data_function, SP=12, suppress_warning=True):
    y_train, y_test, fh = split_data_function
    forecaster = AutoARIMA(sp=SP, suppress_warnings=suppress_warning)
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)
    plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
    print(mean_absolute_error(y_pred, y_test))


def tbats(split_data_function, SP=12, Use_trend=True, Use_box_cox=False):
    y_train, y_test, fh = split_data_function
    forecaster = BATS(sp=SP, use_trend=Use_trend, use_box_cox=Use_box_cox)
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)
    plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
    plt.show()
    print(mean_absolute_error(y_pred, y_test))


def exp_smoothing_MD(data, Trend='add', Seasonal="additive", SP=12):
    forecaster = ExponentialSmoothing(trend=Trend, seasonal=Seasonal, sp=SP)
    forecaster.fit(data)
    y_pred = forecaster.predict(fh)
    plot_series(data, y_pred, labels=["y_train", "y_pred"])
    plt.title("Mindanao")
    plt.xlabel("Date")
    plt.show()
    return y_pred


def ETS_MD(data, Auto=True, SP=12, njobs=-1):
    forecaster = AutoETS(auto=Auto, sp=SP, n_jobs=njobs)
    forecaster.fit(data)
    y_pred = forecaster.predict(fh)
    plot_series(data, y_pred, labels=["y_train", "y_pred"])
    plt.title("Luzon")
    plt.xlabel("Date")
    plt.show()
    return y_pred


def tbats_MD(data, SP=12, Use_trend=True, Use_box_cox=False):
    # y_train, y_test, fh = split_data_function
    forecaster = BATS(sp=SP, use_trend=Use_trend, use_box_cox=Use_box_cox)
    forecaster.fit(data)
    y_pred = forecaster.predict(fh)
    plot_series(data, y_pred, labels=["y_train", "y_pred"])
    plt.title("Visayas")
    plt.xlabel("Date")
    plt.show()
    return y_pred


fh = np.arange(40) + 1
# Luzon_y_train, Luzon_y_test, fh = split_data(Luzon, 40)
# Visayas_train, Visayas_test, fh = split_data(Visayas, 40)
# Mindanao_train, Mindanao_test, fh = split_data(Mindanao, 40)

# ETS(split_data(Luzon, 40))
# tbats(split_data(Visayas, 40))
# exp_smoothing(split_data(Mindanao, 40))

Luzon = ETS_MD(Luzon)
Visayas = tbats_MD(Visayas)
Mindanao = exp_smoothing_MD(Mindanao)
dict = {"Luzon": Luzon, "Visayas": Visayas, "Mindanao": Mindanao}
predictions = pd.DataFrame(dict)
predictions.to_csv("D:/Peak_Demand_Predictions.csv")
