# MLEWorkflow

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

