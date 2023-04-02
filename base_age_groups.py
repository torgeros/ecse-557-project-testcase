import classifier
import matplotlib.pyplot as plt

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
    15: "datasets/base_age_15.csv",
    200: "datasets/base_age_200.csv"
}
print("running on age bins with sizes", dataset_paths.keys())

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
num_iter_avg = 10

for k in dataset_paths:
    print("running on", k)
    acc_sum = 0
    for i in range(num_iter_avg):
        result = classifier.run(dataset_paths[k], onehot_cols, enum_cols, yesno_cols, "Risk_Flag")
        acc_sum += result["accuracy"]
    accuracy_result[k] = acc_sum / num_iter_avg

print(accuracy_result)
# {2: 0.8775178571428572, 3: 0.8764305555555556, 4: 0.876684523809524, 5: 0.8766845238095238, 6: 0.8771111111111111, 7: 0.8769801587301588, 8: 0.8772341269841271, 9: 0.8775674603174602, 10: 0.8770178571428572, 12: 0.8764186507936506, 15: 0.8772261904761904, 200: 0.8769900793650794}

plt.plot(accuracy_result.keys(),accuracy_result.values(), marker='x')
plt.savefig("base_age_groups")
