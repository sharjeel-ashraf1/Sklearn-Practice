# Breast Cancer Classification

Predicting whether a tumor is malignant or benign based on cell measurements, using classification models from scikit-learn.

## Dataset
- **Source:** `sklearn.datasets.load_breast_cancer` (built-in)
- **Size:** 569 rows, 30 numeric features (e.g. mean radius, mean texture, mean smoothness)
- **Target:** 0 = malignant, 1 = benign

## Workflow
1. Loaded data into a pandas DataFrame
2. Split into train/test sets (80/20)
3. Scaled features with `StandardScaler` (fit on train only, applied to test)
4. Trained and compared two models:
   - `LogisticRegression`
   - `RandomForestClassifier`
5. Evaluated using accuracy, confusion matrix, and classification report (precision/recall/f1)

## Results

| Model | Accuracy | False Negatives (missed malignant cases) |
|---|---|---|
| Logistic Regression | 0.974 | 2 |
| Random Forest Classifier | 0.965 | 3 |

**Logistic Regression confusion matrix:**
```
[[41  2]
 [ 1 70]]
```

## What I learned
- Random Forest isn't automatically the better choice — on this dataset, the classes turned out to be fairly linearly separable, so plain Logistic Regression matched or slightly outperformed Random Forest.
- In a medical context, **recall on the malignant class matters more than raw accuracy** — missing a malignant case (a false negative) is a far more serious error than a false alarm. Logistic Regression had fewer false negatives here, which matters more than its small overall accuracy edge.
- This is why trying more than one model and comparing relevant metrics (not just accuracy) is worth doing before picking a final model.

## Tools used
`pandas`, `scikit-learn` (`train_test_split`, `StandardScaler`, `LogisticRegression`, `RandomForestClassifier`, `accuracy_score`, `confusion_matrix`, `classification_report`)
