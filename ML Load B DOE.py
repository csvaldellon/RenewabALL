import pickle

X_user = [2018, "Luzon", "Solar"]

grid = ["Luzon", "Mindanao", "Visayas"]
grid_val = [0, 0, 0]

subtype = ["Biomass", "Geothermal", "Hydro", "Solar", "Wind", "Coal", "Combined", "Diesel", "Gas Turbine", "Natural Gas", "Oil Thermal"]
subtype_val = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for g in grid:
    if X_user[1] == g:
        grid_val[grid.index(g)] = 1
subtype_val[subtype.index(X_user[2])] = 1
X_test = [X_user[0]] + grid_val + subtype_val
print(X_test)

loaded_model = pickle.load(open("ML_C_rf.pkl", "rb"))
result = loaded_model.predict([X_test])
print(result)