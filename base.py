import classifier

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


from sklearn import metrics
import matplotlib.pyplot as plt
y_test = result["y_test"]; y_pred = result["y_pred"]

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()
