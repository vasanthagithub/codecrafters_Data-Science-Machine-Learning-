
import sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from mglearn.datasets import load_extended_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
data = pd.read_csv('D:\codecraft\house-prices.csv')

data['Brick'].replace('Yes', 1, inplace=True)
data['Brick'].replace('No', 0, inplace=True)

data.drop(['Neighborhood'], axis=1, inplace=True)

# Calculate the median of 'Price'
median_price = data['Price'].median()

# Calculate the percentage scale based on median price
data['Percentage_Scale'] = data['Price'].apply(lambda x: (x / median_price) * 50)

# Function to determine 'sold_status' based on the percentage scale
def get_sold_status(percentage):
    if percentage >= 50:
        return 1
    else:
        return 0

# Applying the function to create the 'sold_status' column
data['sold_status'] = data['Percentage_Scale'].apply(get_sold_status)

print(data)

#all features excluding Sold
concepts = np.array(data.iloc[:,0:-1])

#only includes Sold
target = np.array(data.iloc[:,-1])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(concepts, target, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ', accuracy)


# Plotting actual vs predicted prices
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.show()