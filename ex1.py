from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
data = fetch_california_housing()
X_all = data.data
y = data.target
feature_names = data.feature_names

# --- UNIVARIATE REGRESSION (MedInc) ---
X_uni = X_all[:, [0]]  # Selecting 'MedInc'
X_train, X_test, y_train, y_test = train_test_split(X_uni, y, test_size=0.2, random_state=42)

model_uni = LinearRegression().fit(X_train, y_train)
y_pred_uni = model_uni.predict(X_test)

plt.scatter(X_test, y_test, color='red', label="Actual Values")
sorted_idx = np.argsort(X_test[:,0])
plt.plot(X_test[sorted_idx], y_pred_uni[sorted_idx], color='blue', label="Predicted Line")
plt.xlabel("MedInc")
plt.ylabel("House Value")
plt.title("Univariate Regression")
plt.legend()
plt.show()

# --- BIVARIATE REGRESSION (MedInc, HouseAge) ---
X_bi = X_all[:, [0, 1]]
X_train, X_test, y_train, y_test = train_test_split(X_bi, y, test_size=0.2, random_state=42)

model_bi = LinearRegression().fit(X_train, y_train)
y_pred_bi = model_bi.predict(X_test)

print("Bivariate Coefficients:", model_bi.coef_)
print("Intercept:", model_bi.intercept_)

# --- MULTIVARIATE REGRESSION (All features) ---
X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42)
model_multi = LinearRegression().fit(X_train, y_train)
y_pred_multi = model_multi.predict(X_test)

print("Multivariate Coefficients:", model_multi.coef_)
print("Intercept:", model_multi.intercept_)
print("R² Score:", model_multi.score(X_test, y_test))

plt.scatter(y_test, y_pred_multi, alpha=0.5)
plt.xlabel("Actual Value")
plt.ylabel("Predicted Value")
plt.title("Multivariate Regression: Actual vs Predicted")
plt.show()
