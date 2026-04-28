# Ensemble Regression Benchmark

This project presents a comparative study of three powerful gradient boosting algorithms—XGBoost, LightGBM, and CatBoost—applied to a real-world regression problem. The primary objective is to evaluate their predictive performance on the California Housing dataset through systematic hyperparameter optimization and rigorous evaluation metrics.

## Overview

Gradient boosting models have become a cornerstone of modern machine learning due to their exceptional performance on structured data. However, selecting the most suitable model often requires empirical validation rather than theoretical assumptions.

This repository implements:

- Three state-of-the-art ensemble models
- Hyperparameter tuning using GridSearchCV
- Performance evaluation using Mean Squared Error and R² Score
- Visual comparison of model performance
- Feature importance analysis for interpretability

## Dataset

The project uses the California Housing dataset, which contains information such as:

- Median income
- House age
- Average number of rooms
- Population statistics
- Geographic coordinates

The target variable represents the median house value.

## Models Implemented

- **XGBoost Regressor**
- **LightGBM Regressor**
- **CatBoost Regressor**

Each model is trained and optimized independently using a predefined parameter grid.

## Methodology

1. Data Loading and Preprocessing  
   The dataset is loaded using Scikit-learn and split into training and testing subsets.

2. Model Initialization  
   Three gradient boosting models are instantiated with fixed random states for reproducibility.

3. Hyperparameter Optimization  
   GridSearchCV is employed to exhaustively search the parameter space for optimal configurations.

4. Model Evaluation  
   Each model is evaluated on:
   - Mean Squared Error (MSE)
   - R² Score (coefficient of determination)

5. Visualization  
   - Bar plot comparing R² scores across models
   - Feature importance visualization for XGBoost

## Results

The models are compared based on their R² scores. Higher values indicate better predictive capability. The results are visualized using Seaborn for clarity and interpretability.

## Installation

Clone the repository:

```bash
git cloInstall dependencies:
```
```bash
pip install numpy matplotlib seaborn scikit-learn xgboost lightgbm catboost
```
Usage

## Run the script:

python main.py

## The script will:

Train all models
Perform hyperparameter tuning
Output best parameters
Display evaluation metrics
Generate visualizations
## Key Insights
Ensemble methods significantly outperform traditional regression techniques on structured datasets.
Hyperparameter tuning plays a pivotal role in unlocking model performance.
Feature importance provides valuable interpretability, revealing which variables exert the most influence on predictions.
## Future Improvements
Incorporate cross-validation with more folds for robustness
Add more regression models for broader comparison
Implement automated pipelines using tools like Optuna
Deploy the best-performing model as an API
## License

This project is open-source and available under the MIT mark
