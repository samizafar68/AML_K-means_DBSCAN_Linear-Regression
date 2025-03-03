# Applied Machine Learning 

## Overview
This consists of two major tasks:
1. **K-Means & DBSCAN Clustering** - Segmenting customers based on purchasing behavior.
2. **Multiple Linear Regression** - Predicting house prices using multiple regression techniques.

## Requirements
Ensure you have the following dependencies installed before running the code:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

---

# 1) K-Means & DBSCAN Clustering

## Dataset
The dataset consists of customer purchasing behavior with features such as:
- `Annual_Income`
- `Spending_Score`
- `Website_Visits`
- `Product_Categories_Purchased`
- `Total_Purchase_Amount`
- `Average_Session_Duration`
- `Return_Rate`
- `Discount_Usage`

## 1. Load and Preprocess the Dataset
- Handle missing values.
- Normalize numerical features using Z-score standardization.

## 2. Determine the Optimal Number of Clusters (K)
- Use **Elbow Method** to determine the optimal `K`.
- Validate using **Silhouette Score**.

## 3. Apply K-Means Clustering
- Implement **K-Means with optimal K**.
- Visualize clusters using a scatter plot.

## 4. Cluster Interpretation
- Assign cluster labels.
- Analyze average values of each feature for different clusters.

## 5. Handling Outliers
- Detect outliers using the IQR method.
- Reapply clustering after outlier removal.

## 6. Comparison with DBSCAN
- Implement **DBSCAN**.
- Compare **K-Means** and **DBSCAN** clustering results.

---

# 2) Multiple Linear Regression

## Dataset
The dataset consists of housing features:
- `House_Age`
- `Num_Bedrooms`
- `Area_Sqft`
- `Distance_to_City_Center`
- `House_Price` (Target variable)

## 1. Data Preprocessing and Exploration
- Handle missing values.
- Detect and remove outliers.
- Visualize feature distributions.
- Analyze relationships between features and `House_Price`.

## 2. Feature Engineering and Selection
- Normalize numerical features using Z-score standardization.
- Compute feature correlations.
- Create polynomial features to capture non-linearity.

## 3. Train a Linear Regression Model
- Split dataset into training and testing sets.
- Train a **Linear Regression model**.
- Evaluate model using **MAE, MSE, and R² Score**.

## 4. Implement Linear Regression Using Gradient Descent
- Implement manual **Linear Regression using Gradient Descent**.
- Tune hyperparameters (learning rate, epochs).
- Compare with Scikit-learn’s model.

## 5. Predict House Prices for New Data
- Create a function to predict house prices.
- Test model on unseen data.

## How to Run the Code

### Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### Run K-Means & DBSCAN Clustering
```python
Question_no_1_Clustering(K-means,DBSCAN).ipynb
```

### Run Linear Regression
```python
Question_No_2_Linear_Regression.ipynb
```

### Predict House Prices
```python
from predict import predict_house_price
pred_price = predict_house_price(House_Age=93, Num_Bedrooms=1, Area_Sqft=3885, Distance_to_City_Center=36.65)
print(f'Predicted Price: ${pred_price:.2f}')
```

