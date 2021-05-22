import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import linear_model
from sklearn.metrics import r2_score
import pickle
from sklearn.model_selection import GridSearchCV

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
l_reg = linear_model.LinearRegression()  # Build model
cvs_random = np.mean(cross_val_score(l_reg, x_train, y_train, scoring="neg_mean_absolute_error"))
# print(cvs_random)

# 2 Grid Search optimization: will select parameters of model which optimizes model

parameters = {"fit_intercept": [True, False], "normalize": [True, False], "copy_X": [True, False], "positive":
    [True, False], "n_jobs": [1, -1, -2]}
gs = GridSearchCV(l_reg, parameters, scoring="neg_mean_absolute_error")

# METHOD 1: linear regression
# 1 Data modeling

gs.fit(x_train, y_train)      # Train model

pred_l_reg = gs.best_estimator_.predict(x_test)                 # linear regression

# METRICS OF ACCURACY/ERROR
# 1 mean absolute error (from Ken Jee tutorial)
from sklearn.metrics import mean_absolute_error
error_l_reg = mean_absolute_error(y_test, pred_l_reg)
print("error_l_reg:", error_l_reg)

# 2 r^2 score
accuracy_l_reg = r2_score(y_test, pred_l_reg)
print("accuracy_l_reg:", accuracy_l_reg)


# VISUALIZATION OF ACCURACY/ERROR
# 1 data vs prediction plot
plt.scatter(y_test, pred_l_reg)
plt.plot([0, 5e06], [0, 5e06])
plt.xlabel("actual data")
plt.ylabel("prediction")
plt.show()


# 2 error distribution
error_plot_rf = y_test - pred_l_reg
plt.hist(error_plot_rf)
plt.ylabel("count")
plt.xlabel("error")
plt.show()