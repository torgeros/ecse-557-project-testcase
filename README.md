# ECSE 557 Project Test Case

This belongs to a McGill University project. The related dataset modification tool can be found in [elements72/ECSE-557-project](https://github.com/elements72/ECSE-557-project)

## `classifier.py` usage

Warning: enum_str does enumerate just fine, but in the case of binned/grouped values, they are still not ordered...

## Datasets

Datasets can be found in `/datasets`.

| filename      | description |
| ------------- | ----------- |
| full          | original dataset as exported from kaggle |
| base          | blank ID for all entries. all following are based on `base`. Run enum on ID to set to 0 |
| base_debiased | as many zero-targets as one-targets. training bias is too big otherwise. |
| b_d_inc_*     | base_debiased, but with income grouped to filename based bin size. Create using our tool. |

## Classifier python files

Executable python scripts can be found in the root directory, the are named according to the used dataset (see above table).

The `print_b_d_json_accuracies.py` script is used to evaluate the json files logged by the `b_d_inc_*` classifier and create confusion matrices used in the report.

Files that should not be ran directly (because they are internal libraries of sort) are
- `classifier.py`, providing the actual classifier code
- `conf_mat_plt.py`, a utility for creating a confusion matrix with matplotlib

The `helpers` directory contains a script to create the *base_debiased* data set from *base*.
