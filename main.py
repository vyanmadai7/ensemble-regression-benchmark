import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

data = fetch_california_housing()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


models = {
    "XGBoost": XGBRegressor(random_state=42),
    "LGBM": LGBMRegressor(random_state=42),
    "CatBoost": CatBoostRegressor(verbose=0, random_state=42)
}


param_grids = {
    "XGBoost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1]
    },
    "LGBM": {
        "n_estimators": [100, 200],
        "num_leaves": [31, 50],
        "learning_rate": [0.05, 0.1]
    },
    "CatBoost": {
        "iterations": [100, 200],
        "depth": [4, 6],
        "learning_rate": [0.05, 0.1]
    }
}

results = {}

for name in models:
    print(f"\nTraining {name}...")

    grid = GridSearchCV(
        estimator=models[name],          
        param_grid=param_grids[name],    
        cv=3,
        scoring='r2',
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {
        "model": best_model,
        "MSE": mse,
        "R2": r2
    }

    print(f"Best params: {grid.best_params_}")
    print(f"R2 Score: {r2:.4f}")


names = list(results.keys())
r2_scores = [results[m]["R2"] for m in names]

plt.figure(figsize=(8, 5))
sns.barplot(x=names, y=r2_scores)
plt.title("Model Comparison")
plt.ylabel("R2 Score")
plt.show()

xgb_model = results["XGBoost"]["model"]
importances = xgb_model.feature_importances_

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=data.feature_names)
plt.title("XGBoost Feature Importance")
plt.show()
