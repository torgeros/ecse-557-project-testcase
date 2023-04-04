import json
import conf_mat_plt

JSON_PATH = "exports/b_d_income_bins_binned_results_2.json"

with open(JSON_PATH, mode="r") as data_file:
    data = json.load(data_file)

for key in data:
    acc = data[key]["accuracy"]
    print(key, ": accuracy ", acc, sep="")

result = data["0"]
conf_mat_plt.plot(result["y_test"], result["y_pred"])
result = data["10000"]
conf_mat_plt.plot(result["y_test"], result["y_pred"])
