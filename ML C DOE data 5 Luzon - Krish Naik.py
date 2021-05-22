import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler


def plot_peak_demand():
    plt.plot(time, df_3_Luzon_Peak_Demand, label="Luzon")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Peak Demand (MW)")
    plt.show()


def prepare_data(timeseries_data, n_features):  # Thanks to Krish Naik
    X, y = [], []
    for i in range(len(timeseries_data)):
        end_ix = i + n_features
        if end_ix > len(timeseries_data) - 1:
            break
        seq_x, seq_y = timeseries_data[i:end_ix], timeseries_data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


df_3 = pd.read_csv("D:/DS Contests/1 Accenture/4 Datasets by Others/DOE data 3.csv")
df_3 = df_3.drop(columns=["Unnamed: 0"])
df_3["YYYY-MM"] = pd.to_datetime(df_3["YYYY-MM"])


df_3_Luzon = df_3.loc[df_3["Grid"] == "Luzon"].sort_values(by="YYYY-MM")

df_3_Luzon_Peak_Demand = df_3_Luzon["Peak Demand (MW)"].values
time = df_3_Luzon["YYYY-MM"].values


scaler = MinMaxScaler(feature_range=(0, 1))
df_3_Luzon_Peak_Demand = scaler.fit_transform(df_3_Luzon_Peak_Demand.reshape(-1, 1))


def train_test_split(df1):
    training_size = int(len(df1)*0.80)
    test_size = len(df1)-training_size
    train_data, test_data = df1[0:training_size, :], df1[training_size:len(df1), :1]
    return train_data, test_data


def create_dataset(dataset, time_step):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


train_data_Luzon, test_data_Luzon = train_test_split(df_3_Luzon_Peak_Demand)
train_data_Luzon, val_data_Luzon = train_test_split(train_data_Luzon)
timestep = 12
X_train_Luzon, y_train_Luzon = create_dataset(train_data_Luzon, timestep)
X_val_Luzon, y_val_Luzon = create_dataset(val_data_Luzon, timestep)

X_train_Luzon = X_train_Luzon.reshape(X_train_Luzon.shape[0], X_train_Luzon.shape[1], 1)
X_val_Luzon = X_val_Luzon.reshape(X_val_Luzon.shape[0], X_val_Luzon.shape[1], 1)

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(timestep, 1)),
    LSTM(50, return_sequences=True),
    LSTM(50),
    Dense(1)
])
model.compile(loss="mean_squared_error", optimizer="adam")
print(model.summary())

model.fit(X_train_Luzon, y_train_Luzon, validation_split=0.2, epochs=100000000, batch_size=64, verbose=1, shuffle=False)
print(time)
train_predict = model.predict(X_train_Luzon)
val_predict = model.predict(X_val_Luzon)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(val_predict)

look_back = timestep
trainPredictPlot = np.empty_like(df_3_Luzon_Peak_Demand)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
valPredictPlot = np.empty_like(df_3_Luzon_Peak_Demand)
valPredictPlot[:, :] = np.nan
valPredictPlot[len(train_predict)+(look_back*2)+1:len(df_3_Luzon_Peak_Demand)-1, :] = val_predict

plt.plot(time, scaler.inverse_transform(df_3_Luzon_Peak_Demand), label="actual")
plt.plot(time, trainPredictPlot, label="train")
plt.plot(time, valPredictPlot, label="val")
plt.legend()
plt.show()
print(time)
