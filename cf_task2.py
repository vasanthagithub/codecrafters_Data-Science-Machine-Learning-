import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
# For example, assuming you have a CSV file named 'house_data.csv' with columns 'area' and 'price'
# Replace 'house_data.csv' with your dataset file name
data = pd.read_csv('D:\codecraft\house-prices.csv')

# Assuming 'area' as the feature and 'price' as the target variable
X = data['SqFt'].values.reshape(-1, 1)  # Features
y = data['Price'].values  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Fit the model with the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Plotting the graph of predictions between actual price & predicted price
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual Price vs Predicted Price')
plt.show()

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
