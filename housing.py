import pandas as pd
import matplotlib.pyplot as pl
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso

pl.ion()
pf = pd.read_csv('Housing.csv')
print(pf.head())

print(pf.describe())
print(pf.info())

pf.dropna(inplace=True)
#print(pf.describe())
#print(pf.info())
# Distribution of house prices
pl.figure(figsize=(10, 6))
sns.histplot(pf['price'], bins=30, kde=True)
pl.title('Distribution of House Prices (Histplot)')
pl.xlabel('Price')
pl.ylabel('Frequency')
pl.show()
pl.pause(10)
pl.close()

numeric_columns = pf.select_dtypes(include=['float64', 'int64'])
# Correlation heatmap to see relationships between numerical features
pl.figure(figsize=(12, 8))
sns.heatmap(numeric_columns.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
pl.title('Correlation Heatmap')
pl.show()
pl.pause(10)
pl.close()

#pl.figure(figsize=(10, 6))
fig, axes= pl.subplots(nrows=2, ncols=2, figsize=(16, 8))

sns.boxplot(x='basement', y='price', data=pf, ax=axes[0, 0])
axes[0, 0].set_title('House Prices by Number of Bedrooms')
pl.xlabel('basement')
pl.ylabel('Price')


sns.boxplot(x='mainroad', y='price', data=pf, ax=axes[0,1])
axes[0, 1].set_title('House Prices by near the mainroom')
pl.xlabel('mainroad')
pl.ylabel('Price')


sns.boxplot(x='guestroom', y='price', data=pf, ax=axes[1,0])
axes[1,0].set_title('House Prices by avaibility of guestroom')
pl.xlabel('guestroom')
pl.ylabel('Price')


sns.boxplot(x='parking', y='price', data=pf, ax=axes[1,1])
axes[1,1].set_title('House Prices by avaibility of parking')
pl.xlabel('parking')
pl.ylabel('Price')

pl.tight_layout()
pl.show()
pl.pause(10)
pl.close()

pl.figure(figsize=(10, 6))
sns.scatterplot(x='area', y='price', data=pf)
pl.title('area vs. Price (Scatterplot)')
pl.xlabel('area')
pl.ylabel('Price')
pl.show()
pl.pause(10)
pl.close()

pl.figure(figsize=(10, 6))
sns.lineplot(x='area', y='price', data=pf)
pl.title('area vs. Price (Lineplot)')
pl.xlabel('area')
pl.ylabel('Price')
pl.show()
pl.pause(10)
pl.close()

sns.set(style="ticks")  # Set the style of the plot
sns.pairplot(pf, vars=['price', 'area', 'bedrooms', 'bathrooms'])
pl.suptitle('Pair Plot of Housing Data', y=1.02)
pl.show()
pl.pause(30)
pl.close()




# Define features (X) and target (y) for regression
X_reg = pf[['area']]
y_reg = pf['price']

# Split the dataset for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.25)

# Feature Scaling
scaler = StandardScaler()
X_train_reg_scaled = scaler.fit_transform(X_train_reg)
X_test_reg_scaled = scaler.transform(X_test_reg)

# Create a regression model (Linear Regression is used as an example)
regressor = LinearRegression()

# Train the regression model
regressor.fit(X_train_reg_scaled, y_train_reg)

# Make predictions on the test data for regression
y_pred_reg = regressor.predict(X_test_reg_scaled)

# Evaluate the regression model
mse = mean_squared_error(y_test_reg, y_pred_reg)
print(f'Mean Squared Error (Regression): {mse:.2f}')

# Display a scatter plot for regression
pl.figure(figsize=(10, 6))
pl.scatter(X_test_reg, y_test_reg, color='blue', label='Actual Prices')
pl.plot(X_test_reg, y_pred_reg, color='red', linewidth=2, label='Predicted Prices')
pl.title('Regression: Actual vs. Predicted Prices')
pl.xlabel('Area')
pl.ylabel('Price')
pl.legend()
pl.show()
pl.pause(10)
pl.close()


X_cls = pf[['area']]
y_cls = pf['mainroad']  # Assuming 'mainroad' is the binary classification target

# Split the dataset for classification
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.25)

scaler_cls = StandardScaler()
X_train_cls_scaled = scaler_cls.fit_transform(X_train_cls)
X_test_cls_scaled = scaler_cls.transform(X_test_cls)
# Create a classifier (Decision Tree is used as an example)
classifier = DecisionTreeClassifier()

# Train the classifier
classifier.fit(X_train_cls, y_train_cls)

# Make predictions on the test data for classification
y_pred_cls = classifier.predict(X_test_cls)

# Evaluate the classifier
accuracy = accuracy_score(y_test_cls, y_pred_cls)
print(f'Accuracy (Classification): {accuracy:.2f}')

# Display a scatter plot for classification
pl.figure(figsize=(10, 6))
pl.scatter(X_test_cls, y_test_cls, color='blue', label='Actual Classes')
pl.scatter(X_test_cls, y_pred_cls, color='red', marker='x', label='Predicted Classes')
pl.title('Classification: Actual vs. Predicted Classes')
pl.xlabel('Area')
pl.ylabel('Main Road (1: Yes, 0: No)')
pl.legend()
pl.show()
pl.pause(10)
pl.close()

# Regression with Lasso Regression
X_reg = pf[['area']]
y_reg = pf['price']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.25)

scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

regressor_lasso = Lasso(alpha=1.0)  # You can adjust the alpha (regularization parameter) as needed
regressor_lasso.fit(X_train_reg_scaled, y_train_reg)
y_pred_reg_lasso = regressor_lasso.predict(X_test_reg_scaled)
mse_lasso = mean_squared_error(y_test_reg, y_pred_reg_lasso)
print(f'Mean Squared Error (Lasso Regression): {mse_lasso:.2f}')

# Display a scatter plot for regression
pl.figure(figsize=(10, 6))
pl.scatter(X_test_reg, y_test_reg, color='blue', label='Actual Prices')
pl.plot(X_test_reg, y_pred_reg_lasso, color='red', linewidth=2, label='Predicted Prices (Lasso Regression)')
pl.title('Regression: Actual vs. Predicted Prices (Lasso Regression)')
pl.xlabel('Area')
pl.ylabel('Price')
pl.legend()
pl.show()
pl.pause(10)
pl.close()

# Classification with Random Forest
X_cls = pf[['area']]
y_cls = pf['mainroad']
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.25)

scaler_cls = StandardScaler()
X_train_cls_scaled = scaler_cls.fit_transform(X_train_cls)
X_test_cls_scaled = scaler_cls.transform(X_test_cls)

classifier_rf = RandomForestClassifier()
classifier_rf.fit(X_train_cls_scaled, y_train_cls)
y_pred_cls_rf = classifier_rf.predict(X_test_cls_scaled)
accuracy_rf = accuracy_score(y_test_cls, y_pred_cls_rf)
print(f'Accuracy (Random Forest Classification): {accuracy_rf:.2f}')

# Display a scatter plot for classification
pl.figure(figsize=(10, 6))
pl.scatter(X_test_cls, y_test_cls, color='blue', label='Actual Classes')
pl.scatter(X_test_cls, y_pred_cls_rf, color='red', marker='x', label='Predicted Classes (Random Forest)')
pl.title('Classification: Actual vs. Predicted Classes (Random Forest)')
pl.xlabel('Area')
pl.ylabel('Main Road (1: Yes, 0: No)')
pl.legend()
pl.show()
pl.pause(10)
pl.close()

