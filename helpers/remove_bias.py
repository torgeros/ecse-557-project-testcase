
# this script assures that numbers of 0 <= number of 1

FULL_DATASET_PATH = "../datasets/full.csv"
DEBIASED_DATASET_PATH = "../datasets/full_debiased.csv"

with open(FULL_DATASET_PATH, "r") as f:
    label = f.readline()
    lines = f.readlines()

print(lines[:10])

count_target_1 = 0
for l in lines:
    if (l[-2] == "1"):
        count_target_1 += 1
    elif (l[-2] == "0"):
        pass
    else:
        print("UM THERE WAS A CHAR", l[-2])

new_lines = [label]
print(new_lines)
count_placed_0s = 0
for l in lines:
    if (l[-2] == "1"):
        new_lines.append(l)
    elif (l[-2] == "0"):
        if count_placed_0s <= count_target_1:
            new_lines.append(l)
            count_placed_0s += 1

with open(DEBIASED_DATASET_PATH, "w") as f:
    f.writelines(new_lines)
