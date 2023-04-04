import classifier
import conf_mat_plt

DATASET_PATH = "datasets/base_debiased.csv"
FIGURE_EXPORT_PATH = "exports/b_d_inc_figures"
JSON_EXPORT_PATH = 'exports/b_d_income_bins_binned_results.json'

binned_datasets = {
      100: "b_d_inc_100k.csv",
      200: "b_d_inc_200k.csv",
      500: "b_d_inc_500k.csv",
      750: "b_d_inc_750k.csv",
     1000: "b_d_inc_1000k.csv",
     1250: "b_d_inc_1250k.csv",
     1500: "b_d_inc_1500k.csv",
     2000: "b_d_inc_2000k.csv",
     3000: "b_d_inc_3000k.csv",
     5000: "b_d_inc_5000k.csv",
    10000: "b_d_inc_10000k.csv"
}

onehot_cols_full = [
    "Married/Single",
    "House_Ownership",
    "Profession",
    "CITY",
    "STATE"
]

onehot_cols = onehot_cols_full + ["Income"]

enum_cols = [
    "Id"
]

yesno_cols = [
    "Car_Ownership"
]

print("CLASSIFICATION RESULTS:")

full_result = classifier.run(DATASET_PATH, onehot_cols_full, enum_cols, yesno_cols, "Risk_Flag", should_print=False)
print("full base_d accuracy:", full_result["accuracy"])

binned_results = dict()
binned_results[0] = full_result

for group_size in binned_datasets:
    filename = "datasets/" + binned_datasets[group_size]
    binned_results[group_size] = classifier.run(filename, onehot_cols, enum_cols, yesno_cols, "Risk_Flag")
    print("bin size ", group_size, ":", sep="")
    print("\taccuracy:", binned_results[group_size]["accuracy"])

for gs in binned_results:
    # <class 'pandas.core.series.Series'>
    binned_results[gs]["y_test"] = binned_results[gs]["y_test"].values.tolist()
    # <class 'numpy.ndarray'>
    binned_results[gs]["y_pred"] = binned_results[gs]["y_pred"].tolist()

import json
with open(JSON_EXPORT_PATH, 'w') as f:
    json.dump(binned_results, f)
