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



from sklearn import metrics
import matplotlib.pyplot as plt
y_test = result["y_test"]; y_pred = result["y_pred"]

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()
