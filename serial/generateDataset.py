from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Generate a regression dataset
X, y = make_regression(n_samples=50000, n_features=8, n_informative=8, noise=0.1, random_state=42)

# Initialize the StandardScaler
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Fit the scaler to the features and transform them
X_standardized = scaler_X.fit_transform(X)

# Fit the scaler to the target and transform it
y_standardized = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Create a DataFrame from the standardized features
df = pd.DataFrame(X_standardized, columns=[f'Feature_{i}' for i in range(1, 9)])

# Add the standardized target variable to the DataFrame
df['Target'] = y_standardized

# Save the DataFrame to a CSV file without headers or index
df.to_csv('newyork1.csv', index=False, header=False)
