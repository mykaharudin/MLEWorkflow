# House Price Prediction Machine learning

dataset: (https://www.kaggle.com/datasets/syuzai/perth-house-prices)

## Data Collection in Machine Learning House PRice

Data collection is the foundation of machine learning. Without relevant and high-quality data, models cannot learn or make accurate predictions.

### Key Points:
- **Accuracy**: Better data, better results.
- **Bias Prevention**: Balanced data avoids bias.
- **Diverse Data**: Varied data improves adaptability.

## "NEXT DATA Load"
# Loading Dataset in Machine Learning

Loading a dataset is the first and essential step in the machine learning process. It involves fetching data from external sources and importing it into the working environment for further analysis. Once loaded, you can begin exploring, cleaning, and training your machine learning model, all of which depend on the quality and readiness of the data.

## Data Cleaning
Before building a predictive model, it's crucial to clean the dataset by handling missing values and ensuring that all columns are ready for analysis.

### Identifying Missing Values
We start by identifying missing values in the dataset. The columns with missing values in the dataset are:
- `NEAREST_SCH_RANK`: School ranking, some schools are not ranked.
- `GARAGE`: Number of garages.
- `BUILD_YEAR`: Year the house was built.

### Handling Missing Values
To address the missing values, the following strategies were applied:

- **GARAGE and BUILD_YEAR**: Filled with the median value since both are numerical features with missing values that don't have a strong pattern.
  
  ```python
  data['GARAGE'].fillna(data['GARAGE'].median(), inplace=True)
  data['BUILD_YEAR'].fillna(data['BUILD_YEAR'].median(), inplace=True)

# Data Exploration and Feature Engineering
1. [Feature Engineering](#feature-engineering)
   - [HOUSE_AGE (Umur Rumah)](#house_age)
   - [Analysis of HOUSE_AGE](#analysis-of-house_age)
2. [Exploration of Key Features](#exploration-of-key-features)
   - [BEDROOMS (Jumlah Kamar Tidur)](#bedrooms)
   - [BATHROOMS (Jumlah Kamar Mandi)](#bathrooms)
   - [CBD_DIST (Jarak ke Pusat Kota)](#cbd_dist)

##  Feature Exploration
We explored and engineered several key features used to predict house prices:
- **HOUSE_AGE**: The age of the house had **weak correlation** with house prices.
- **BEDROOMS**: There was a slight correlation, where houses with more bedrooms tended to be a bit more expensive.
- **BATHROOMS**: **Number of bathrooms** showed a stronger correlation with house prices; houses with more bathrooms were generally more expensive.
- **CBD_DIST (Distance to City Center)**: Houses **closer to the city center** tended to have higher prices.

## Models Tested
We tried several machine learning models to predict house prices:
- **Linear Regression**: This model provided less accurate results, with a **Mean Absolute Error (MAE)** of **209,760 AUD**.
- **Random Forest**: Improved compared to Linear Regression, with an MAE of **193,990 AUD**.
- **Gradient Boosting**: Gave better results, with an MAE of **188,236 AUD**.
- **XGBoost**: The **best model** with an MAE of **182,542 AUD**, making it the most accurate model for predicting house prices.

## Recommendations
- **XGBoost** is the **best model** for predicting house prices in this dataset, with the lowest MAE. This model is suitable as it can handle more complex patterns in the data.
- The most influential features on house prices are:
  - **Number of bathrooms (BATHROOMS)**
  - **Distance to city center (CBD_DIST)**
  - **Number of bedrooms (BEDROOMS)**
- **HOUSE_AGE** had a smaller impact but can still be used to support the predictions.

## Next Steps
To further improve performance:
- You can add additional features or further optimize the model using techniques such as **hyperparameter tuning** or **cross-validation** atau other ways that are more ***OPTIMAL***. 
"""