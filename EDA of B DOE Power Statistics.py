import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv("D:/DOE Power Statistics 2003-2020 Installed and Dependable Capacity per grid by Plant Type.csv")

# print(data.head())
# print(data.columns)
# print(data.info())
# print(data.describe())

df_num = data[["Year", "Capacity"]]
df_cat = data[["Installed or Dependable", "Region", "Plant Type", "Plant Subtype"]]

# for i in df_num.columns:
    # plt.hist(df_num[i])
    # plt.title(i)
    # plt.show()

# print(df_num.corr())
# sns.heatmap(df_num.corr())
# plt.show()

region = pd.pivot_table(data, index="Region", values="Capacity")
region.plot(kind="bar")

plant_type = pd.pivot_table(data, index="Plant Type", values="Capacity")
plant_type.plot(kind="bar")

plant_subtype = pd.pivot_table(data, index="Plant Subtype", values="Capacity")
plant_subtype.rename(index={"None": "Non-renewable"}).plot(kind="bar")
inst_or_dep = pd.pivot_table(data, index="Installed or Dependable", values="Capacity")
inst_or_dep.plot(kind="bar")
plt.show()


# sns.barplot(data=[region.index, region.values]).set_title("region")
# plt.show()

# QUESTIONS: Given several factors (location, year, plant type, plant substype), can you predict the capacity of a
# power plant?
