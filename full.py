import classifier

DATASET_PATH = "datasets/full.csv"

# all str_cols are actually containing enumerating strings
str_cols = [
    "Married/Single",
    "House_Ownership",
    "Profession",
    "CITY",
    "STATE"
]

# translate yes/no col to boolean value
yesno_cols = [
    "Car_Ownership"
]

result = classifier.run(DATASET_PATH, str_cols, [], yesno_cols, "Risk_Flag", should_print=True)
print("CLASSIFICATION RESULTS:")
print(result.keys())
