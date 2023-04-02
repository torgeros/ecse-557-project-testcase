import classifier

dataset_paths = {
    2: "datasets/base_age_2.csv",
    3: "datasets/base_age_3.csv",
    4: "datasets/base_age_4.csv",
    5: "datasets/base_age_5.csv",
    6: "datasets/base_age_6.csv",
    7: "datasets/base_age_7.csv",
    8: "datasets/base_age_8.csv",
    9: "datasets/base_age_9.csv",
    10: "datasets/base_age_10.csv",
    12: "datasets/base_age_12.csv",
    15: "datasets/base_age_15.csv"
}
print("running on age bins with sizes", dataset_paths)

onehot_cols = [
    "Age",
    "Married/Single",
    "House_Ownership",
    "Profession",
    "CITY",
    "STATE"
]

enum_cols = [
    "Id"
]

# translate yes/no col to boolean value
yesno_cols = [
    "Car_Ownership"
]

accuracy_result = dict()

for k in dataset_paths:
    result = classifier.run(dataset_paths[k], onehot_cols, enum_cols, yesno_cols, "Risk_Flag", should_print=True)
    accuracy_result[k] = result["accuracy"]

print(accuracy_result)
# {2: 0.8762103174603174, 3: 0.8774404761904762, 4: 0.8749404761904762, 5: 0.8758928571428571, 6: 0.8770634920634921, 7: 0.8775198412698413, 8: 0.8766269841269841, 9: 0.8757142857142857, 10: 0.8758333333333334, 12: 0.8778769841269841, 15: 0.8774801587301587}
