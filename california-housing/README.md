# California Housing Price Prediction

Predicting median house values in California districts using regression models from scikit-learn.

## Dataset
- **Source:** `sklearn.datasets.fetch_california_housing` (built-in)
- **Size:** 20,640 rows, 8 features
- **Features:** median income, house age, average rooms, average bedrooms, population, average occupancy, latitude, longitude
- **Target:** median house value (in $100,000s)

## Workflow
1. Loaded data into a pandas DataFrame
2. Split into train/test sets (80/20)
3. Scaled features with `StandardScaler` (fit on train only, applied to test)
4. Trained two models and compared results:
   - `LinearRegression`
   - `RandomForestRegressor`
5. Evaluated using MSE and R²

## Results

| Model | MSE | R² |
|---|---|---|
| Linear Regression | 0.528 | 0.608 |
| Random Forest Regressor | 0.273 | 0.797 |

## What I learned
- Linear Regression assumes a straight-line relationship between features and price, which doesn't hold well here — especially for `latitude`/`longitude`, where price depends on location in complex, non-linear ways (e.g. proximity to the coast).
- Random Forest doesn't assume any fixed relationship shape, since it splits data recursively on thresholds. This let it capture the non-linear patterns and pushed R² from ~0.61 to ~0.80.
- Tree-based models like Random Forest don't require feature scaling, since they split on thresholds rather than distances — unlike Linear Regression, which does benefit from scaling.

## Tools used
`pandas`, `scikit-learn` (`train_test_split`, `StandardScaler`, `LinearRegression`, `RandomForestRegressor`, `mean_squared_error`, `r2_score`)
