""" 
--------------------------------------------------------------
    Import Libraries
--------------------------------------------------------------
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


""" 
--------------------------------------------------------------
    Download Economic Data Using yFinance
--------------------------------------------------------------
"""
# Download stock market data (e.g., S&P 500 as a proxy for the overall market)
sp500_data = yf.download('^GSPC', start='2000-01-01', end='2023-01-01')
sp500_data['Return'] = sp500_data['Adj Close'].pct_change()  # Calculate daily returns

# Download bond yield data (as an economic indicator)
bond_yield_data = yf.download('^TNX', start='2000-01-01', end='2023-01-01')  # 10-year Treasury yield
bond_yield_data['Yield'] = bond_yield_data['Adj Close']

# Merge datasets by date
data = sp500_data[['Return']].merge(bond_yield_data[['Yield']], left_index=True, right_index=True)
data.dropna(inplace=True)  # Remove rows with missing values


""" 
--------------------------------------------------------------
    Define a Financial Crisis Event
--------------------------------------------------------------
"""
# Define a crisis if the return drops below a certain threshold (e.g., -5% in one day)
data['Crisis'] = (data['Return'] < -0.05).astype(int)

# Check the distribution of crisis events
print(data['Crisis'].value_counts())


# Visualize the Data
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(data.index, data['Return'], label='S&P 500 Returns')
plt.axhline(y=-0.05, color='r', linestyle='--', label='Crisis Threshold (-5%)')
plt.legend()
plt.title('S&P 500 Daily Returns')

plt.subplot(2, 1, 2)
plt.plot(data.index, data['Yield'], label='10-Year Treasury Yield')
plt.title('10-Year Treasury Yield')
plt.tight_layout()
plt.show()


""" 
--------------------------------------------------------------
    Train/Test Split
--------------------------------------------------------------
"""
# Define features (economic indicators) and target (Crisis label)
X = data[['Return', 'Yield']]
y = data['Crisis']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


""" 
--------------------------------------------------------------
    Build the Machine Learning Model
--------------------------------------------------------------
"""
# Build a Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


""" 
--------------------------------------------------------------
    Evaluate the Model
--------------------------------------------------------------
"""
# Make predictions on the test set
y_pred = model.predict(X_test)

# Display confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Plot the predicted vs actual crises
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test.values, label='Actual Crisis Events', color='r', linestyle='--')
plt.plot(y_test.index, y_pred, label='Predicted Crisis Events', color='b')
plt.title('Actual vs Predicted Financial Crises')
plt.legend()
plt.show()


""" 
--------------------------------------------------------------
    Feature Importance Analysis
--------------------------------------------------------------
"""
# Get feature importance
importances = model.feature_importances_
feature_names = X.columns

# Plot feature importance
plt.figure(figsize=(8, 4))
sns.barplot(x=importances, y=feature_names)
plt.title('Feature Importance in Predicting Financial Crises')
plt.show()