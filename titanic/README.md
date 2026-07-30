# Titanic Survival Prediction

Predicting whether a passenger survived the Titanic disaster, using a real-world, messy dataset (missing values, mixed data types) with scikit-learn.

## Dataset
- **Source:** `Titanic-Dataset.csv` (Kaggle Titanic dataset)
- **Size:** 891 rows, 12 original columns
- **Target:** `Survived` (0 = did not survive, 1 = survived)

## Data cleaning
Unlike the built-in datasets, this required real preprocessing decisions:
- **Dropped columns** not useful for prediction: `PassengerId`, `Name`, `Ticket`, `Cabin` (Cabin was missing for 687 of 891 rows)
- **Filled missing values:**
  - `Age` → filled with median
  - `Embarked` → filled with mode (most common value)
- **Encoded categorical columns:** `Sex` and `Embarked` converted to numeric with `pd.get_dummies(..., drop_first=True)`

## Workflow
1. Cleaned and encoded the data as above
2. Split into train/test sets (80/20)
3. Scaled features with `StandardScaler`
4. Trained a `LogisticRegression` model
5. Evaluated using accuracy, confusion matrix, and classification report

## Results

**Accuracy: 0.810**

```
[[90 15]
 [19 55]]
```

|  | Precision | Recall | F1 |
|---|---|---|---|
| Not Survived (0) | 0.83 | 0.86 | 0.84 |
| Survived (1) | 0.79 | 0.74 | 0.76 |

## What I learned
- Real datasets need real cleanup — dropping irrelevant/mostly-missing columns, filling missing values (median for numeric, mode for categorical), and encoding text categories into numbers before any model can be trained.
- Tested whether scaling changed results for Logistic Regression — accuracy stayed identical. Scaling's main benefit here was faster/more reliable convergence, not a different final decision boundary, since the model was given enough iterations (`max_iter=1000`) to converge either way.
- Reading the confusion matrix carefully matters: 19 passengers who actually survived were predicted as not surviving (false negatives), and 15 who didn't survive were predicted as surviving (false positives) — the recall for "Survived" (0.74) is noticeably lower than for "Not Survived" (0.86), showing the model is more conservative about predicting survival.

## Tools used
`pandas`, `scikit-learn` (`train_test_split`, `StandardScaler`, `LogisticRegression`, `accuracy_score`, `confusion_matrix`, `classification_report`)
