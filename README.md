#  House Price Predictor

A machine learning web app that predicts house sale prices based on property features like living area, quality, garage capacity, and more — built end-to-end from raw data to a live, deployed application.

**🔗 Live app:** [https://house-price-app-g35ntlqffpxwgr2ggcb6nj.streamlit.app/]

---

## Overview

This project uses the [Kaggle House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) dataset to train a regression model that predicts a house's sale price from its characteristics. The model is deployed as an interactive web app using Streamlit, so anyone can enter property details and get a price prediction instantly.

## Tech stack

- **Python** — pandas, NumPy, scikit-learn
- **Model** — Linear Regression (with log-transformed target)
- **Deployment** — Streamlit + Streamlit Community Cloud
- **Version control** — Git & GitHub

## Project workflow

1. **Data loading & exploration** — 1460 rows, 81 columns of housing data
2. **Data cleaning**
   - Dropped columns with excessive missing values (`PoolQC`, `MiscFeature`, `Alley`, `Fence`)
   - Filled categorical missing values with `"None"` where absence is meaningful (e.g. no garage, no basement)
   - Filled numeric missing values with median (`LotFrontage`) or mode (`Electrical`)
3. **Feature encoding** — converted 39 categorical columns into numeric format using One-Hot Encoding (`pd.get_dummies`), expanding to 248 columns
4. **Train-test split** — 80/20 split (1168 training rows, 292 test rows)
5. **Model training** — Linear Regression on the processed features
6. **Model improvement** — applied a log transform (`log1p`) to the target variable (`SalePrice`) to correct for right-skewed distribution, which significantly improved performance
7. **Deployment** — saved the trained model with `joblib`, built a Streamlit interface, and deployed via Streamlit Community Cloud

## Results

| Version | R² Score | RMSE |
|---|---|---|
| Baseline Linear Regression | 0.641 | $52,496 |
| Feature-scaled | 0.641 | $52,496 |
| **Log-transformed target** | **0.913** | **$25,790** |

The log transformation gave the biggest performance gain — reducing prediction error by roughly half and explaining 91% of the variance in sale prices, compared to 64% in the baseline.

## Key features used

Based on correlation with `SalePrice`, the app collects these inputs from the user:

- Overall Quality
- Above-ground living area
- Garage capacity
- Total basement area
- 1st floor area
- Full bathrooms
- Year built
- Lot area
- Total rooms above ground
- Fireplaces

Any feature not exposed in the UI defaults to its median value from the training data, so predictions stay realistic even with partial input.

## Running locally

```bash
# clone the repo
git clone https://github.com/<your-username>/house-price-app.git
cd house-price-app

# install dependencies
pip install -r requirements.txt

# run the app
streamlit run app.py
```

## Project structure

```
house-price-app/
├── app.py                    # Streamlit web app
├── house_price_model.pkl     # Trained Linear Regression model
├── model_columns.pkl         # Column order expected by the model
├── default_values.pkl        # Median values used for unspecified features
├── requirements.txt          # Python dependencies
└── README.md
```

## What I'd improve next

- Try Ridge/Lasso regression or tree-based models (Random Forest, XGBoost) for comparison
- Add more input features to the UI for finer-grained predictions
- Perform proper cross-validation instead of a single train-test split
- Add feature importance visualization to the app

## Author

Built as part of a personal ML learning roadmap, applying Linear Regression to a real-world dataset from data cleaning through deployment.
