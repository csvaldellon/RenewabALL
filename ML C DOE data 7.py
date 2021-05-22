import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df_3 = pd.read_csv("D:/DS Contests/1 Accenture/4 Datasets by Others/DOE data 3.csv")
df_3 = df_3.drop(columns=["Unnamed: 0"])
df_3["YYYY-MM"] = pd.to_datetime(df_3["YYYY-MM"])


df_3_Luzon = df_3.loc[df_3["Grid"] == "Luzon"].sort_values(by="YYYY-MM")
time = df_3_Luzon["YYYY-MM"]
# print(len(time))

df_3_Luzon_Peak_Demand = df_3_Luzon.drop(columns="YYYY-MM")
df_3_Luzon_Peak_Demand = df_3_Luzon_Peak_Demand.drop(columns="Grid")

future_days = 40
df_3_Luzon_Peak_Demand["Prediction"] = df_3_Luzon_Peak_Demand[["Peak Demand (MW)"]].shift(-future_days)
# print(df_3_Luzon_Peak_Demand)
X = np.array(df_3_Luzon_Peak_Demand.drop(["Prediction"], 1))[:-future_days]
# print(len(X))
y = np.array(df_3_Luzon_Peak_Demand["Prediction"])[:-future_days]
# print(len(y))

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)

from sklearn.linear_model import SGDRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
from lightgbm import LGBMRegressor

lgbm = LGBMRegressor().fit(x_train, y_train)
kr = KernelRidge().fit(x_train, y_train)
en = ElasticNet().fit(x_train, y_train)
br = BayesianRidge().fit(x_train, y_train)
svr = SVR().fit(x_train, y_train)

tree = DecisionTreeRegressor().fit(x_train, y_train)
lr = LinearRegression().fit(x_train, y_train)
cvs = np.mean(cross_val_score(lr, x_train, y_train, scoring="neg_mean_absolute_error"))
print(cvs)
rf = RandomForestRegressor().fit(x_train, y_train)
# lasso = Lasso(alpha=1).fit(x_train, y_train)
# csv_lasso = np.mean(cross_val_score(lasso, x_train, y_train, scoring="neg_mean_absolute_error"))
# print(csv_lasso)
lasso = Lasso(alpha=1).fit(x_train, y_train)
csv_lasso = np.mean(cross_val_score(lasso, x_train, y_train, scoring="neg_mean_absolute_error"))
print(csv_lasso)
xgb = GradientBoostingRegressor().fit(x_train, y_train)


def find_alpha():
    alpha = []
    error = []
    b = 8e-10
    for i in range(1, 100):
        alpha.append(b * i)
        lml = Lasso(alpha=(b * i))
        error.append(np.mean(cross_val_score(lml, x_train, y_train, scoring="neg_mean_absolute_error")))

    plt.plot(alpha, error)
    plt.show()


parameters = {"n_estimators": range(10, 100, 10), "criterion": ("mse", "mae"), "max_features": ("auto", "sqrt")}
# gs = GridSearchCV(rf, parameters, scoring="neg_mean_absolute_error").fit(x_train, y_train)

x_future = df_3_Luzon_Peak_Demand.drop(["Prediction"], 1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)
# print(len(x_future))
# print(len(x_test))

lgbm_prediction = lgbm.predict(x_future)
kr_prediction = kr.predict(x_future)
en_prediction = en.predict(x_future)
br_prediction = br.predict(x_future)
svr_prediction = svr.predict(x_future)

tree_prediction = tree.predict(x_future)
lr_prediction = lr.predict(x_future)
lasso_prediction = lasso.predict(x_future)
xgb_prediction = xgb.predict(x_future)
rf_prediction = rf.predict(x_future)
# gs_rf_prediction = gs.best_estimator_.predict(x_future)
avg_prediction = (0*lasso_prediction + 2*lr_prediction + xgb_prediction)/3
avg_prediction_2 = (0*lasso_prediction + 2*lr_prediction + rf_prediction)/3
# print(x_test[-45:-35])
# print(x_future[-45:-35])


def plot_year_vs_PD(method, title):
    valid = df_3_Luzon_Peak_Demand[X.shape[0]:]
    valid["Predictions"] = method
    # print(time[len(time) - future_days:])
    plt.plot(time, df_3_Luzon_Peak_Demand.loc[:, "Peak Demand (MW)"], label="Actual")
    # plt.plot(time[len(time)-future_days:], valid["Peak Demand (MW)"])
    plt.plot(time[len(time) - future_days:], valid["Predictions"], label="Predictions")
    plt.legend()
    plt.title(title)
    plt.show()
    rmse = mean_squared_error(valid["Peak Demand (MW)"], valid["Predictions"])
    mae = mean_absolute_error(valid["Peak Demand (MW)"], valid["Predictions"])
    # print(title, " rmse: ", rmse)
    print(title, " mae: ", mae)
    # print(y_test)
    # print(valid["Peak Demand (MW)"])


# plot_year_vs_PD(tree_prediction, "Decision Tree Regression")
plot_year_vs_PD(lgbm_prediction, "lgbm Regression")
plot_year_vs_PD(kr_prediction, "KR Regression")
plot_year_vs_PD(en_prediction, "EN Regression")
plot_year_vs_PD(br_prediction, "BR Regression")
plot_year_vs_PD(svr_prediction, "SVR Regression")
# plot_year_vs_PD(avg_prediction, "Average of 2xOLS and XGB")
# plot_year_vs_PD(avg_prediction_2, "Average of 2xOLS and RF")
