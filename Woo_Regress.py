import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare data
data = pd.read_csv('cleaned_woo.csv')

# Select features and target
features = [
    "release_speed", 
    "release_pos_x", 
    "release_pos_z",
    "release_pos_y",
    "effective_speed",
    "release_spin_rate",
    "release_extension",
    "pitch_name"  # Categorical feature
]

target = "launch_speed"

# Remove rows with missing values
data = data[features + [target]].dropna()


# Split data
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
numeric_features = [f for f in features if f != "pitch_name"]
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_features = ["pitch_name"]
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Modeling pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

# Train model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f} mph")

# Feature Importance Analysis
# Extract feature names after one-hot encoding
onehot_columns = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(['pitch_name'])
all_feature_names = numeric_features + list(onehot_columns)

# Get importances
importances = model.named_steps['regressor'].feature_importances_
importance_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances})
importance_df = importance_df.sort_values('Importance', ascending=False)

# Plot top features
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
plt.title('Top Features Impacting Exit Velocity')
plt.tight_layout()
plt.show()
import shap

# Apply preprocessing to X_train
X_train_processed = model.named_steps['preprocessor'].transform(X_train)

# Explain model predictions using SHAP
explainer = shap.TreeExplainer(model.named_steps['regressor'])
shap_values = explainer.shap_values(X_train_processed)

# Plot summary
shap.summary_plot(shap_values, X_train_processed, feature_names=all_feature_names)
