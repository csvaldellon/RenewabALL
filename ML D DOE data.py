import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import linear_model
from sklearn.metrics import r2_score
import pickle

# Load data
df = pd.read_csv("D:/DS Contests/1 Accenture/1 Datasets Contribution/DOE data D.csv")

# Data pre-processing 1: In general, model accepts subtype. It would accept plant type if no subtype available.
# Q: What if accepts Plant Type only?
for i in np.arange(0, 800, step=10):
    df.loc[i, "Plant Subtype"] = "Coal"
for i in np.arange(4, 804, step=10):
    df.loc[i, "Plant Subtype"] = "Natural Gas"

# Data pre-processing 2: dropping columns that do not represent a feature.
df_model = df.drop(columns=["Plant Type", "Unnamed: 0"])

# Data pre-processing 3: one-hot encoding categorical features so that ML can understand them
df_dum = pd.get_dummies(df_model)

# Data pre-processing 4: filling null values with zero (refer to original dataset for interpretation)
df_dum["Gross Power Generated"] = df_dum["Gross Power Generated"].fillna(0)

# Data pre-processing 5: splitting data into features and labels
X = df_dum.drop("Gross Power Generated", axis=1)  # features
y = df_dum["Gross Power Generated"].values        # labels

# Train-test splitting: about 80% of data will be used to train ML and 20% to test it
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# METHOD 1: linear regression
# 1 Data modeling
l_reg = linear_model.LinearRegression()  # Build model
model = l_reg.fit(x_train, y_train)      # Train model

# 2 Cross validation: out of the 80% of train set, about 20% (so 16% of orig set) will be used to validate the remainder
# using a metric (negative mean absolute error).
cvs = np.mean(cross_val_score(l_reg, x_train, y_train, scoring="neg_mean_absolute_error"))
# print(cvs)


# METHOD 2: Lasso regression
# 1 Data modeling
lasso = linear_model.Lasso(alpha=24400)  # Build model: alpha determined from below
lasso.fit(x_train, y_train)              # Train model

# 2 Cross validation
cvs_lasso = np.mean(cross_val_score(lasso, x_train, y_train, scoring="neg_mean_absolute_error"))
# print(cvs_lasso)

# 3 Determination of optimal alpha (modified from Ken Jee tutorial)

# alpha = []
# error = []

# for i in range(1, 100):
    # alpha.append(400*i)
    # lml = linear_model.Lasso(alpha=(400*i))
    # error.append(np.mean(cross_val_score(lml, x_train, y_train, scoring="neg_mean_absolute_error")))

# plt.plot(alpha, error)
# plt.show()

# rr = tuple(zip(alpha, error))
# df_err = pd.DataFrame(err, columns = ["alpha", "error"])
# print(df_err[df_err.error == max(df_err.error)])


# METHOD 3: Random forest regression
# 1 Data modeling
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
cvs_random = np.mean(cross_val_score(rf, x_train, y_train, scoring="neg_mean_absolute_error"))
# print(cvs_random)

# 2 Grid Search optimization: will select parameters of model which optimizes model
from sklearn.model_selection import GridSearchCV
parameters = {"n_estimators": range(10, 100, 10), "criterion": ("mse", "mae"), "max_features": ("auto", "sqrt", "log2")}
gs = GridSearchCV(rf, parameters, scoring="neg_mean_absolute_error")

# 3 Training of model using grid searched parameters
gs.fit(x_train, y_train)
# print(gs.best_score_)
# print(gs.best_estimator_)

# Data modeling 5: save model
filename = 'ML_D_rf.sav'
pickle.dump(gs, open(filename, 'wb'))

# Prediction
pred_l_reg = l_reg.predict(x_test)                 # linear regression
pred_lasso = lasso.predict(x_test)                 # Lasso regression
pred_rf = gs.best_estimator_.predict(x_test)       # RF regression

# METRICS OF ACCURACY/ERROR
# 1 mean absolute error (from Ken Jee tutorial)
from sklearn.metrics import mean_absolute_error
error_l_reg = mean_absolute_error(y_test, pred_l_reg)
error_lasso = mean_absolute_error(y_test, pred_lasso)
error_rf = mean_absolute_error(y_test, pred_rf)
print("error_l_reg:", error_l_reg)
print("error_lasso:", error_lasso)
print("error_rf:", error_rf)

# 2 r^2 score
accuracy_l_reg = r2_score(y_test, pred_l_reg)
accuracy_lasso = r2_score(y_test, pred_lasso)
accuracy_rf = r2_score(y_test, pred_rf)
print("accuracy_l_reg:", accuracy_l_reg)
print("accuracy_lasso:", accuracy_lasso)
print("accuracy_rf:", accuracy_rf)


# VISUALIZATION OF ACCURACY/ERROR
# 1 data vs prediction plot
plt.scatter(y_test, pred_l_reg)
plt.plot([0, 5e06], [0, 5e06])
plt.xlabel("actual data")
plt.ylabel("prediction")
plt.show()
plt.scatter(y_test, pred_lasso)
plt.plot([0, 5e06], [0, 5e06])
plt.xlabel("actual data")
plt.ylabel("prediction")
plt.show()
plt.scatter(y_test, pred_rf)
plt.plot([0, 5e06], [0, 5e06])
plt.xlabel("actual data")
plt.ylabel("prediction")
plt.show()

# 2 error distribution
error_plot_rf = y_test - pred_rf
plt.hist(error_plot_rf)
plt.ylabel("count")
plt.xlabel("error")
plt.show()
