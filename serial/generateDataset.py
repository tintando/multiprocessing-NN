from sklearn.datasets import make_regression
import pandas as pd

X, y = make_regression(n_samples=50000, n_features=8, n_informative=8, random_state=42)

df = pd.DataFrame(X)
df['Target'] = y

df.to_csv('newyork.csv', index=False, header=False)