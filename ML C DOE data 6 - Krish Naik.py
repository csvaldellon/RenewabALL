import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

df_3 = pd.read_csv("D:/DS Contests/1 Accenture/4 Datasets by Others/DOE data 3.csv")
df_3 = df_3.drop(columns=["Unnamed: 0"])
df_3["YYYY-MM"] = pd.to_datetime(df_3["YYYY-MM"])


df_3_Luzon = df_3.loc[df_3["Grid"] == "Luzon"].sort_values(by="YYYY-MM")
df_3_Visayas = df_3.loc[df_3["Grid"] == "Visayas"].sort_values(by="YYYY-MM")
df_3_Mindanao = df_3.loc[df_3["Grid"] == "Mindanao"].sort_values(by="YYYY-MM")

df_3_Luzon_Peak_Demand = df_3_Luzon["Peak Demand (MW)"].values
df_3_Visayas_Peak_Demand = df_3_Visayas["Peak Demand (MW)"].values
df_3_Mindanao_Peak_Demand = df_3_Mindanao["Peak Demand (MW)"].values
time = df_3_Luzon["YYYY-MM"].values


def plot_peak_demand():
    plt.plot(time, df_3_Luzon_Peak_Demand, label="Luzon")
    plt.plot(time, df_3_Visayas_Peak_Demand, label="Visayas")
    plt.plot(time, df_3_Mindanao_Peak_Demand, label="Mindanao")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Peak Demand (MW)")
    plt.show()

