# ECSE 557 Project Test Case

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
