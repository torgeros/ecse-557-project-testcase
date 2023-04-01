import classifier

DATASET_PATH = "datasets/01.csv"

onehot_cols = [
    "Married/Single",
    "House_Ownership",
    "Profession",
    "CITY",
    "STATE"
]

enum_cols = [
    "Id",
    "Income",
    "Age"
]

# translate yes/no col to boolean value
yesno_cols = [
    "Car_Ownership"
]

result = classifier.run(DATASET_PATH, onehot_cols, enum_cols,  yesno_cols, "Risk_Flag", True)
print("CLASSIFICATION RESULTS:")
print(result.keys())
