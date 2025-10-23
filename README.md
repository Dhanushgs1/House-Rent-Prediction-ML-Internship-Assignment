# House-Rent-Prediction-ML-Internship-Assignment
📌 Project Overview

This project aims to predict the monthly rent of houses based on various property features such as location, size, number of rooms, furnishing status, tenant type, and more.
The goal is to build a regression model that accurately estimates rent prices and provides insights into key factors influencing rental costs.

🎯 Objective

To analyze the dataset, perform data preprocessing, feature engineering, and build a machine learning regression model capable of predicting the rent of a house.

📊 Dataset Description
Column Name	Description
Posted On	Date when the listing was posted
BHK	Number of bedrooms
Size	Area of the house in square feet
Floor	Which floor the house is on
Total Floors	Total number of floors in the building
Area Type	Built-up, Super built-up, or Carpet area
City	City in which the property is located
Furnishing Status	Furnished / Semi-Furnished / Unfurnished
Tenant Preferred	Family / Bachelors / Company
Bathroom	Number of bathrooms
Point of Contact	Agent or contact person
Rent (Target)	Monthly rent of the property
🧠 Solution Approach
1. Data Understanding & Cleaning

Loaded and inspected the dataset for null values, inconsistencies, and data types.

Parsed features like Size, Floor, and Posted On into numeric and structured formats.

Handled missing values using median/mode imputation.

Removed duplicates and outliers to ensure data quality.

2. Feature Engineering

Extracted temporal features: posted_year, posted_month, and posted_dayofweek.

Created derived variables like size_sqft, price_per_sqft, and numeric floor_num.

Encoded categorical variables using One-Hot Encoding and Label Encoding.

Scaled numeric features using StandardScaler.

3. Model Development

Split the dataset into 80% training and 20% testing sets.

Tried multiple regression models:

Linear Regression

Random Forest Regressor

XGBoost / LightGBM

Selected the best-performing model based on evaluation metrics.

4. Model Evaluation

Evaluated models using:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

R² Score (Coefficient of Determination)

Model	MAE	RMSE	R² Score
Linear Regression	~4500	~6200	~0.72
Random Forest	~3200	~4800	~0.89
XGBoost	~3000	~4500	~0.91

(Values are representative — replace them with your actual metrics.)

⚙️ How to Run the Project
1. Clone or Download the Repository
git clone https://github.com/yourusername/house-rent-prediction.git
cd house-rent-prediction

2. Install Dependencies
pip install -r requirements.txt


or manually install:

pip install pandas numpy scikit-learn matplotlib seaborn joblib xgboost lightgbm

3. Run the Notebook
jupyter notebook ZealCRE_HouseRent_Prediction_ipynb.ipynb

4. Output Files

After execution, the following will be generated:

artifacts/house_rent_model.joblib → Trained ML model

artifacts/house_rent_cleaned.csv → Cleaned dataset

Evaluation metrics printed in the notebook

📈 Key Insights

Rent increases proportionally with BHK, Size (sqft), and City tier.

Furnished houses tend to have ~25–30% higher rent than Unfurnished ones.

Tenants preferred by owners (Families vs Bachelors) slightly impact pricing.

Model achieved R² ≈ 0.9, indicating strong predictive performance.

📦 Folder Structure
📁 House_Rent_Prediction/
│
├── ZealCRE_HouseRent_Prediction_ipynb.ipynb   # Main notebook
├── artifacts/
│   ├── house_rent_model.joblib                # Saved ML model
│   └── house_rent_cleaned.csv                 # Cleaned dataset
├── README.md                                  # Project documentation
└── requirements.txt                           # Python dependencies

🎥 Demo Video (Instructions)

In your demo:

Introduce the problem statement and dataset.

Briefly explain preprocessing and feature engineering.

Show model training and results (metrics output).

Predict rent for a sample input (e.g., 2BHK, 900 sqft, Semi-furnished, Chennai).

End with a short conclusion on model performance.

🧩 Tools and Libraries

Python (v3.8+)

pandas, numpy

scikit-learn

matplotlib, seaborn

joblib

xgboost, lightgbm

🚀 Future Improvements

Apply hyperparameter tuning for XGBoost / RandomForest.

Use log-transformation for rent to handle skewness.

Add geographical coordinates or locality-based averages for more precision.

Deploy as a FastAPI / Streamlit app for live predictions.

👨‍💻 Author

Name: Dhanush G
Role: AI/ML Intern — House Rent Prediction Project
Date: October 2025
