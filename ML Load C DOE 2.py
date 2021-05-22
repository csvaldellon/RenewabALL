from flask import Flask, render_template, request
import pickle


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = 0
    if request.method == "POST":
        user_year = request.form.get("year")
        user_grid = request.form.get("grid")
        user_subtype = request.form.get("subtype")
        X_user = [user_year, user_grid, user_subtype]
        prediction = get_predict(X_user)[0]
        prediction = round(prediction)
    return render_template("RenewabALL_predict_C.html", prediction=prediction)


# @app.route("/predict", methods=["GET", "POST"])
def get_predict(X_user):
    grid = ["Luzon", "Mindanao", "Visayas"]
    grid_val = [0, 0, 0]

    subtype = ["Biomass", "Geothermal", "Hydro", "Solar", "Wind", "Coal", "Combined", "Diesel", "Gas Turbine",
               "Natural Gas", "Oil Thermal"]
    subtype_val = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for g in grid:
        if X_user[1] == g:
            grid_val[grid.index(g)] = 1
    subtype_val[subtype.index(X_user[2])] = 1
    X_test = [X_user[0]] + grid_val + subtype_val
    # print(X_test)

    loaded_model = pickle.load(open("ML_C_rf.pkl", "rb"))
    result = loaded_model.predict([X_test])
    return result


if __name__ == "__main__":
    app.run(debug=True)
