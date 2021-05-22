import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
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

# Train-test splitting: about 80% of data will be used to train ML and 20% to test it
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.50)

# Data modeling 1: build model
from sklearn.ensemble import RandomForestRegressor  # import packages
rf = RandomForestRegressor()                        # build model

# Data modeling 2: cross validation - out of the 80% of train set, about 20% (so 16% of orig set) will be used to
# validate the remainder using a metric (negative mean absolute error)
cvs_random = np.mean(cross_val_score(rf, x_train, y_train, scoring="neg_mean_absolute_error"))
# print(cvs_random)

# Data modeling 3: Grid search cross validation - optimizes parameters which minimizes metric of cross validation
from sklearn.model_selection import GridSearchCV
parameters = {"n_estimators": range(10, 100, 10), "criterion": ("mse", "mae"), "max_features": ("auto", "sqrt", "log2")}
gs = GridSearchCV(rf, parameters, scoring="neg_mean_absolute_error")

# Data modeling 4: training of model using grid searched parameters
gs.fit(x_train, y_train)

# Data modeling 5: save model
filename = 'ML_C_rf.pkl'
pickle.dump(gs.best_estimator_, open(filename, 'wb'))

# Prediction
pred_rf = gs.best_estimator_.predict(x_test)

# Metrics of accuracy/error
from sklearn.metrics import mean_absolute_error
error_rf = mean_absolute_error(y_test, pred_rf)
print("error_rf:", error_rf)
accuracy_rf = r2_score(y_test, pred_rf)
print("accuracy_rf:", accuracy_rf)

# Visualization of accuracy/error
plt.scatter(y_test, pred_rf)
plt.plot([0, 4.1e07], [0, 4.1e07])
plt.xlabel("actual data")
plt.ylabel("prediction")
plt.show()

error_plot_rf = y_test - pred_rf
plt.hist(error_plot_rf)
plt.ylabel("count")
plt.xlabel("error")
plt.show()
