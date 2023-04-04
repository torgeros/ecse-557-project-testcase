import classifier
import conf_mat_plt

DATASET_PATH = "datasets/base.csv"

onehot_cols = [
    "Married/Single",
    "House_Ownership",
    "Profession",
    "CITY",
    "STATE"
]

enum_cols = [
    "Id"
]

yesno_cols = [
    "Car_Ownership"
]

result = classifier.run(DATASET_PATH, onehot_cols, enum_cols, yesno_cols, "Risk_Flag", should_print=False)
print("CLASSIFICATION RESULTS:")
print("accuracy:", result["accuracy"])

conf_mat_plt.plot(result["y_test"], result["y_pred"])
