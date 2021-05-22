import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score

# Load data
df = pd.read_csv("D:/DS Contests/1 Accenture/1 Datasets Contribution/DOE data B.csv")

# Data pre-processing 1: In general, model accepts subtype. It would accept plant type if no subtype available.
# Q: What if accepts Plant Type only?
for i in np.arange(0, 864, step=8):
    df.loc[i, "Plant Subtype"] = "Coal"
for i in np.arange(1, 864, step=8):
    df.loc[i, "Plant Subtype"] = "Oil Based"
for i in np.arange(2, 864, step=8):
    df.loc[i, "Plant Subtype"] = "Natural Gas"

# Data pre-processing 2: dropping columns that do not represent a feature.
df_model = df.drop(columns="Plant Type")

# Data pre-processing 3: one-hot encoding categorical features so that ML can understand them
df_dum = pd.get_dummies(df_model)

# Data pre-processing 5: splitting data into features and labels
X = df_dum.drop("Capacity", axis=1)     # features
y = df_dum["Capacity"].values           # labels

# Train-test splitting 1: about 20% (about 173 points) of data will be used to train ML and 80% (about 691 points)
# to FINALLY test it
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.80)

# Train-test splitting 2: I will use the training set from above as the new dataset to build the model with
x_1 = x_train
y_1 = y_train
x_test_final = x_test
y_test_final = y_test

# Train-test splitting 3: about 80% (about 138 points) of data will be used to train ML and 20% (about 35 points)
# to INITIALLY test it
x_train, x_test, y_train, y_test = train_test_split(x_1, y_1, test_size=0.20)

# Data modeling 1: build model
from sklearn.ensemble import RandomForestRegressor  # import packages
rf = RandomForestRegressor()                        # build model

# Data modeling 2: cross validation - out of the 80% of train set, about 20% will be used to validate the remainder
# using a metric (negative mean absolute error)
cvs_random = np.mean(cross_val_score(rf, x_train, y_train, scoring="neg_mean_absolute_error"))
# print(cvs_random)

# Data modeling 3: Grid search cross validation - optimizes parameters which minimizes metric of cross validation
from sklearn.model_selection import GridSearchCV
parameters = {"n_estimators": range(10, 100, 10), "criterion": ("mse", "mae"), "max_features": ["auto", "sqrt", "log2"],
              "max_depth": [750, 800, 950, 900], "min_samples_split": [2, 5], "min_samples_leaf": [1, 2], "bootstrap": [True, False]}
gs = GridSearchCV(rf, parameters, scoring="neg_mean_absolute_error", cv=10, verbose=2, n_jobs=4)

# Data modeling 4: training of model using grid searched parameters
gs.fit(x_train, y_train)

print(gs.best_estimator_)

# Prediction 1: testing of model
pred_rf = gs.best_estimator_.predict(x_test)
pred_rf_final = gs.best_estimator_.predict(x_test_final)

# Prediction 2: metrics of accuracy/error with initial testing set
from sklearn.metrics import mean_absolute_error
error_rf = mean_absolute_error(y_test, pred_rf)
print("error_rf:", error_rf)
accuracy_rf = r2_score(y_test, pred_rf)
print("accuracy_rf:", accuracy_rf)

# Prediction 3: visualization of accuracy/error with initial testing set
plt.scatter(y_test, pred_rf)
plt.plot([0, 7100], [0, 7100])
plt.xlabel("actual data")
plt.ylabel("prediction")
plt.show()

error_plot_rf = y_test - pred_rf
plt.hist(error_plot_rf)
plt.ylabel("count")
plt.xlabel("error")
plt.show()

# Prediction 4: metrics of accuracy/error with FINAL testing set
error_rf_final = mean_absolute_error(y_test_final, pred_rf_final)
print("error_rf_final:", error_rf_final)
accuracy_rf_final = r2_score(y_test_final, pred_rf_final)
print("accuracy_rf_final:", accuracy_rf_final)

# Prediction 5: visualization of accuracy/error with FINAL testing set
plt.scatter(y_test_final, pred_rf_final)
plt.plot([0, 7100], [0, 7100])
plt.xlabel("actual data final")
plt.ylabel("prediction final")
plt.show()

error_plot_rf_final = y_test_final - pred_rf_final
plt.hist(error_plot_rf_final)
plt.ylabel("count final")
plt.xlabel("error final")
plt.show()
