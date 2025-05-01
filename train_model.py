import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np

# --- 1. Load Data ---
print("Loading data...")
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# --- 2. Initial Data Cleaning ---
print("Initial data cleaning...")
# Drop the 'id' column as it's not useful for prediction
df = df.drop('id', axis=1)

# Handle 'Other' gender if necessary (rare case, let's remove for simplicity here)
# Or you could map it or use more robust encoding
df = df[df['gender'] != 'Other']

# Convert BMI to numeric, coercing errors (like 'N/A') to NaN
df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')

# --- 3. Define Features (X) and Target (y) ---
print("Defining features and target...")
X = df.drop('stroke', axis=1)
y = df['stroke']

# --- 4. Split Data ---
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 5. Preprocessing ---
print("Setting up preprocessing pipelines...")

# Identify column types
numerical_features = ['age', 'avg_glucose_level', 'bmi']
# Binary features can sometimes be treated as categorical or handled separately
# Let's use OneHotEncoder for all categorical for consistency here
categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
binary_features = ['hypertension', 'heart_disease'] # These are already 0/1

# Create preprocessing pipelines for different column types

# Pipeline for numerical features: Impute missing values (median) then scale
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Use median for BMI NaNs
    ('scaler', StandardScaler())
])

# Pipeline for categorical features: Impute missing (rare) then One-Hot Encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Handle potential rare NaNs
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # Ignore categories not seen in training
])

# No transformation needed for binary features as they are already 0/1
# If they were 'Yes'/'No', we'd map them or include in categorical

# Create a ColumnTransformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
        # Pass through binary features directly
        ('binary', 'passthrough', binary_features)
        # Note: The order matters for reconstructing feature names if needed later
    ],
    remainder='passthrough' # Keep any other columns (though we shouldn't have any)
)

# --- 6. Define Model ---
print("Defining the model (Logistic Regression)...")
# Using Logistic Regression as a simple starting point
model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced') # Balanced weights for imbalanced data

# --- 7. Create Full Pipeline (Preprocessing + Model) ---
print("Creating the full prediction pipeline...")
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# --- 8. Train the Model ---
print("Training the pipeline...")
pipeline.fit(X_train, y_train)

# --- 9. Evaluate the Model (Optional but Recommended) ---
print("Evaluating the model...")
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- 10. Save the Pipeline and Training Columns ---
print("Saving the trained pipeline and training columns...")
joblib.dump(pipeline, 'stroke_prediction_pipeline.joblib')

# Save the columns used for training (important for ensuring prediction input matches)
training_columns = list(X_train.columns)
joblib.dump(training_columns, 'training_columns.joblib')

# Also save the median BMI used for imputation (needed for single prediction preprocessing)
# We need to fit the imputer *only* on the training data's BMI column to get the median
bmi_imputer = SimpleImputer(strategy='median')
bmi_imputer.fit(X_train[['bmi']])
median_bmi = bmi_imputer.statistics_[0]
joblib.dump(median_bmi, 'median_bmi.joblib')


print("\n--- Training and saving complete! ---")
print("Files created: stroke_prediction_pipeline.joblib, training_columns.joblib, median_bmi.joblib")