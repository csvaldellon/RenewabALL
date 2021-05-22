import pandas as pd
import numpy as np


df = pd.read_csv("D:/DS Contests/1 Accenture/1 Datasets Contribution/DOE data C.csv")

for i in np.arange(0, 592, step=11):
    df.loc[i, "Plant Subtype"] = "Coal"
for i in np.arange(5, 592, step=11):
    df.loc[i, "Plant Subtype"] = "Natural Gas"

df_model = df.drop(columns=["Plant Type", "Unnamed: 0"])
df_dum = pd.get_dummies(df_model)
df_dum["Gross Power Generated"] = df_dum["Gross Power Generated"].fillna(0)

X = df_dum.drop("Gross Power Generated", axis=1)
y = df_dum["Gross Power Generated"].values

# print(X)
# print(len(X))
# print(len(y))

train_size = int(len(X)*0.9)
test_size = len(X) - train_size
x_train = X.iloc[0:train_size]
x_test = X.iloc[train_size:len(X)]
y_train = y[0:train_size]
y_test = y[train_size:len(X)]

# print(len(x_train))
# print(len(x_test))

# print(x_train.shape)
x_train = x_train.to_numpy()
# x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
# x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
# print(x_train.shape)
# print(x_train)
x_new = []
for x in x_train:
    x_new += [[x]]
x_train = np.array(x_new)
# print(x_train)
# print(x_train.shape)
# print(type(x_train[0][0][2]))
# print(type(y_train[0]))
x_train = x_train.astype("float64")


def build_model():
    from tensorflow import keras
    model = keras.Sequential()
    model.add(
        keras.layers.Bidirectional(
            keras.layers.LSTM(
                units=128,
                input_shape=(1, 15)
            )
        )
    )
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.Dense(units=1))

    model.compile(loss="mean_squared_error", optimizer="adam")
    history = model.fit(
        x_train, y_train,
        epochs=1000,
        batch_size=10,
        validation_split=0.1,
        shuffle=False,
        verbose=2
    )


build_model()
