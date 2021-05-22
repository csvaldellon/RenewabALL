import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pickle

# Load data
df = pd.read_csv("D:/DS Contests/1 Accenture/1 Datasets Contribution/DOE data C.csv")

# Data pre-processing 1: In general, model accepts subtype. It would accept plant type if no subtype available.
# Q: What if accepts Plant Type only?
for i in np.arange(0, 592, step=11):
    df.loc[i, "Plant Subtype"] = "Coal"
for i in np.arange(5, 592, step=11):
    df.loc[i, "Plant Subtype"] = "Natural Gas"

# Data pre-processing 2: dropping columns that do not represent a feature.
df_model = df.drop(columns=["Plant Type", "Unnamed: 0"])

# Data pre-processing 3: one-hot encoding categorical features so that ML can understand them
df_dum = pd.get_dummies(df_model)

# Data pre-processing 4: filling null values with zero (refer to original dataset for interpretation)
df_dum["Gross Power Generated"] = df_dum["Gross Power Generated"].fillna(0)

# Data pre-processing 5: splitting data into features and labels
X = df_dum.drop("Gross Power Generated", axis=1)     # features
y = df_dum["Gross Power Generated"].values           # labels

print(X)
print(y)
# Train-test splitting: about 80% of data will be used to train ML and 20% to test it
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.50)


ratio_up = 2015
ratio_down = 2009

X_before_ratio = X.loc[X["Year"] < ratio_down]
y_before_ratio = y[X["Year"] < ratio_down]
X_after_ratio = X.loc[ratio_down <= X["Year"] <= ratio_up]
y_after_ratio = y[ratio_down <= X["Year"] <= ratio_up]

x_train = X_before_ratio
x_test = X_after_ratio
y_train = y_before_ratio
y_test = y_after_ratio


def get_predict():
    from sklearn.ensemble import GradientBoostingRegressor
    xg = GradientBoostingRegressor()
    from sklearn.model_selection import GridSearchCV
    parameters = {"loss": ["ls", "lad", "huber", "quantile"], "learning_rate": np.arange(0.1, 1, 0.1),
                  "n_estimators": range(10, 100, 10)}
    gs = GridSearchCV(xg, parameters, scoring="neg_mean_absolute_error")
    gs.fit(x_train, y_train)
    filename = 'ML_C_xg.pkl'
    pickle.dump(gs.best_estimator_, open(filename, 'wb'))
    pred = gs.best_estimator_.predict(x_test)
    return pred


pred_xg = get_predict()

# Metrics of accuracy/error
from sklearn.metrics import mean_absolute_error
error_rf = mean_absolute_error(y_test, pred_xg)
print("error_rf:", error_rf)
accuracy_rf = r2_score(y_test, pred_xg)
print("accuracy_rf:", accuracy_rf)

# Visualization of accuracy/error
plt.scatter(y_test, pred_xg)
plt.plot([0, 4.1e07], [0, 4.1e07])
plt.legend()
plt.xlabel("actual data")
plt.ylabel("prediction")
plt.show()

error_plot_rf = y_test - pred_xg
plt.hist(error_plot_rf)
plt.legend()
plt.ylabel("count")
plt.xlabel("error")
plt.show()

# result = pd.DataFrame([years, y_test, pred_rf])
# result.to_csv("D:/result C.csv")
