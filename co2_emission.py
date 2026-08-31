import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

# =============================================
# CHANGE THIS PATH TO YOUR CSV FILE LOCATION
# =============================================
DATA_PATH = r'D:\ML\CO2 emission\data\CO2 Emissions.csv'

# Create plots folder
os.makedirs('plots', exist_ok=True)

# --- 1. Load Data ---
df = pd.read_csv(DATA_PATH)
print("✅ Data loaded. Columns:", df.columns.tolist())

# --- 2. Define Features & Target ---
X = df.drop('CO2 Emissions(g/km)', axis=1)
y = df['CO2 Emissions(g/km)']

# --- 3. Preprocessing ---
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

# --- 4. Train/Test & Model ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- 5. Metrics ---
print("\n" + "="*50)
print("LINEAR REGRESSION METRICS")
print("="*50)
print(f"R² Score:  {r2_score(y_test, y_pred):.4f}")
print(f"MAE:       {mean_absolute_error(y_test, y_pred):.4f}")
print(f"RMSE:      {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
cv = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print(f"CV R² (mean): {cv.mean():.4f} (+/- {cv.std():.4f})")

# --- 6. GRAPH 1: Actual vs Predicted ---
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue', edgecolors='black', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect')
plt.xlabel('Actual CO2 (g/km)')
plt.ylabel('Predicted CO2 (g/km)')
plt.title('REGRESSION GRAPH: Actual vs Predicted')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('plots/task1_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/task1_actual_vs_predicted.png")

# --- 7. GRAPH 2: Residuals ---
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.6, color='green', edgecolors='black', linewidth=0.5)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Predicted CO2 (g/km)')
plt.ylabel('Residuals (g/km)')
plt.title('REGRESSION GRAPH: Residuals Plot')
plt.grid(True, alpha=0.3)
plt.savefig('plots/task1_residuals.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/task1_residuals.png")

print("\n🎉 Task 1 COMPLETE! All graphs in 'plots/' folder.")                       