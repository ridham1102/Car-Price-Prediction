# Car-Price-Prediction

## Project Overview

This project aims to develop a robust machine learning model to accurately predict the selling prices of used cars in the dynamic Indian market. Leveraging a comprehensive dataset scraped from CarDekho, the analysis delves into various factors influencing used car valuations, including vehicle age, mileage, fuel type, transmission type, engine specifications, and more. The project follows a standard machine learning workflow, encompassing data loading, extensive preprocessing, model selection, training, and rigorous evaluation of multiple regression algorithms. The ultimate goal is to identify the most effective model for price prediction, providing valuable insights for both buyers and sellers in the used car market.

## Data Collection and Initial Inspection

The dataset used in this project was obtained by scraping the CarDekho website, a popular platform for buying and selling used cars in India. The raw dataset initially contained over 1.4 million rows and 13 columns, providing a rich source of information on individual car listings.

Upon loading the dataset into a pandas DataFrame, an initial inspection was performed to understand its structure and content. The `df.head()` function was used to display the first few rows, giving a glimpse of the columns and the type of data they hold. Key columns identified include:

*   `car_name`: The full name of the car (e.g., 'Maruti Alto', 'Hyundai Grand').
*   `brand`: The brand or manufacturer of the car (e.g., 'Maruti', 'Hyundai').
*   `model`: The specific model of the car (e.g., 'Alto', 'Grand', 'i20').
*   `vehicle_age`: The age of the vehicle in years.
*   `km_driven`: The total distance the car has been driven in kilometers.
*   `seller_type`: The type of seller (e.g., 'Individual', 'Dealer').
*   `fuel_type`: The type of fuel the car uses (e.g., 'Petrol', 'Diesel', 'CNG').
*   `transmission_type`: The transmission type (e.g., 'Manual', 'Automatic').
*   `mileage`: The fuel efficiency of the car (often in km/l).
*   `engine`: The engine capacity (often in CC).
*   `max_power`: The maximum power output of the engine (often in bhp).
*   `seats`: The number of seating capacity in the car.
*   `selling_price`: The target variable, representing the selling price of the used car.

The `df.info()` method provided a summary of the DataFrame, including the number of non-null entries in each column and their respective data types. This step confirmed that there were no missing values across any of the columns in the loaded dataset, simplifying the data cleaning process. The data types were primarily integers (`int64`), floating-point numbers (`float64`), and objects (`object`) for categorical features.

Finally, `df.isnull().sum()` was used to explicitly verify the absence of missing values, confirming the initial observation from `df.info()`.

## Data Preprocessing and Feature Engineering

The raw data required several preprocessing steps to be suitable for training machine learning models.

One of the initial steps involved dropping the 'brand' column. While 'car_name' and 'model' provide more specific information about the vehicle, the 'brand' column was considered redundant for the prediction task, as the brand information is implicitly captured within the 'model' and 'car_name'.

The 'model' column, despite being categorical, has a relatively high number of unique values (120). To handle this effectively while preserving the ordinal nature that might exist in some car models' hierarchy or market positioning, label encoding was applied to the 'model' column. This transformed the categorical model names into numerical labels.

Categorical features such as `seller_type`, `fuel_type`, and `transmission_type` are nominal in nature, meaning there is no inherent order between the categories. For these features, one-hot encoding was employed. This technique converts each category into a new binary column, preventing the models from misinterpreting these categories as having an ordinal relationship. The `drop='first'` argument was used in the `OneHotEncoder` to avoid multicollinearity.

Numerical features in the dataset, including `vehicle_age`, `km_driven`, `mileage`, `engine`, `max_power`, and `seats`, have different scales. To ensure that no single feature dominates the learning process due to its magnitude, `StandardScaler` was applied to these numerical columns. Standardization scales the features to have zero mean and unit variance.

The preprocessing steps were orchestrated using `ColumnTransformer`. An initial `ColumnTransformer` was set up to apply one-hot encoding to the specified categorical columns and to explicitly drop the 'car_name' column, which was not needed for modeling. The `remainder='passthrough'` argument ensured that all other columns (including the label-encoded 'model' and the numerical features) were retained. A second `ColumnTransformer` was then used to apply `StandardScaler` specifically to the numerical columns that were passed through from the initial transformation. This two-step process ensured that categorical and numerical features were handled appropriately and in the correct sequence.

The processed data, now consisting entirely of numerical features, was then split into training and testing sets using `train_test_split` from `sklearn.model_selection`. A test size of 25% was used, and a `random_state` was set for reproducibility.

## Model Training and Evaluation

With the data preprocessed, several regression models were trained to predict the `selling_price`. The models selected for evaluation were:

*   **Random Forest Regressor:** An ensemble learning method that constructs multiple decision trees and outputs the average of their predictions. Known for its robustness and ability to handle non-linear relationships.
*   **Ridge Regression:** A linear regression model that uses L2 regularization to prevent overfitting.
*   **Lasso Regression:** A linear regression model that uses L1 regularization, which can also perform feature selection by shrinking some coefficients to zero.
*   **K-Nearest Neighbors Regressor:** A non-parametric model that predicts the value of a new data point based on the average of its k nearest neighbors in the training data.
*   **Decision Tree Regressor:** A tree-like model that splits the data based on features to make predictions. Can be prone to overfitting.

A custom function, `evaluate_model`, was defined to streamline the training and evaluation process for each model. This function takes a model instance, the training features and target, and the testing features and target as input. It trains the model on the training data, generates predictions on both the training and testing sets, and calculates the following evaluation metrics:

*   **Mean Squared Error (MSE):** The average of the squared differences between the actual and predicted values.
*   **Mean Absolute Error (MAE):** The average of the absolute differences between the actual and predicted values.
*   **Root Mean Squared Error (RMSE):** The square root of the MSE, providing an error metric in the same units as the target variable.
*   **R2 Score:** The coefficient of determination, representing the proportion of the variance in the target variable that is predictable from the features. A higher R2 score indicates a better fit.

The `evaluate_model` function returns a dictionary containing these metrics for both the training and testing sets.

Each of the selected regression models was then initialized and passed to the `evaluate_model` function along with the scaled training and testing data. The results for each model were stored in a dictionary called `model_performance`.

## Results and Analysis

The performance metrics for each model on both the training and testing sets were displayed in a transposed pandas DataFrame for easy comparison. The results showed significant variations in performance across the different algorithms:

*   **Random Forest:** Demonstrated strong performance, with a high R2 score on the test set (0.923) and a relatively low test RMSE (around 214,927). This indicates that the model is able to explain a large portion of the variance in selling prices and provides reasonably accurate predictions.
*   **KNN:** Also performed well on the test set, with an R2 score of 0.919 and a test RMSE of around 219,900, comparable to the Random Forest.
*   **Ridge and Lasso:** As linear models, they struggled to capture the non-linear relationships in the data, resulting in much lower R2 scores (around 0.685 on the test set) and higher RMSE values (around 433,000).
*   **Decision Tree:** Achieved a very high R2 score on the training set (0.999), indicating significant overfitting. Its performance dropped substantially on the unseen test data (R2 of 0.633 and RMSE of around 467,969), highlighting its inability to generalize well.

Based on the evaluation metrics, the **Random Forest Regressor** and **K-Nearest Neighbors Regressor** emerged as the most promising models for predicting used car prices in this dataset. While Random Forest showed a slightly better R2 score on the test set, further hyperparameter tuning could potentially improve the performance of both models.

## Conclusion

This project successfully implemented and evaluated several regression models for used car price prediction in India. The analysis highlighted the importance of proper data preprocessing, including handling categorical features and scaling numerical data. The Random Forest and KNN models proved to be the most effective among those tested, providing a solid foundation for building a predictive tool for the used car market. Future work could involve exploring more advanced feature engineering techniques, hyperparameter tuning of the best-performing models, and potentially investigating other ensemble methods or deep learning approaches to further enhance prediction accuracy.
