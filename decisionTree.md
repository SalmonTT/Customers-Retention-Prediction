# Decision Tree Classifier Tunning with GridSearchCV

### Currently use GridSearchCV to find the optimal config for the following parameters:
```python
params = {
        "criterion": ("gini", "entropy"),
        "splitter": ("best", "random"),
        "max_depth": (list(range(1, 10))),
        "min_samples_split": list(range(10, 30)),
        "min_samples_leaf": list(range(1, 20)),
    }
```
Results obtained so far is as follows:
```
Train Result:
===========================================
accuracy score: 0.8643

Classification Report: 
 	Precision: 0.7747747747747747
	Recall Score: 0.48509266720386784
	F1 score: 0.5966303270564915

Confusion Matrix: 
 [[4584  175]
 [ 639  602]]

Test Result:
===========================================
accuracy score: 0.8506666666666667

Classification Report: 
 	Precision: 0.7326732673267327
	Recall Score: 0.46540880503144655
	F1 score: 0.5692307692307692

Confusion Matrix: 
 [[1128   54]
 [ 170  148]]
```
Note that the confusion matrix is organized like so in Python:

|   | Predicted Negative  |Predicted Positive   |
|---|---|---|
| **Actual Negative**  |  True Negative | True Positive  |
|  **Actual Positive** |  False Negative | True Positive  |

Therefore we see that the problems with current model is as follows:
1. *Low Recall*: 
    - Out of all the actual positive records, we can only predict around **50%** to be positive
2. *Precision could be improved*:
    - Test results hovers at low **70%** precision, which is relatively low

## Things to try next:
1. Better data pre-processing
    - this is less important as decision tree models are less reliant on data
2. Ensemble methods:
    - Bagging and Boosting
    - https://towardsdatascience.com/decision-tree-ensembles-bagging-and-boosting-266a8ba60fd9
3. Techniques described in the documentation:
    - https://scikit-learn.org/stable/modules/tree.html#tree 