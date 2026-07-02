import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

def main():
    print("Loading dataset...")
    # -------------------------------------------------------------------
    # A. DATA CLEANING
    # -------------------------------------------------------------------
    try:
        # Load data
        df = pd.read_csv(r"C:\xampp\htdocs\DjangoStartMasraMIT209\mit209projMasra\ml\dataset\student-lifestyle-and-stress-dataset.csv")
    except FileNotFoundError:
        print("Error: 'student-lifestyle-and-stress-dataset.csv' not found.")
        return

    # 1. Remove duplicates
    df = df.drop_duplicates()

    # 2. Force conversion to numeric for safety (skipping the known text column)
    for col in df.columns:
        if col != 'Student_Type':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. NULL Treatment (Without using 'inplace=True' which can cause bugs)
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if col == 'Student_Type' or df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())

    # 4. Categorical Encoding
    le = LabelEncoder()
    if 'Student_Type' in df.columns:
        df['Student_Type'] = le.fit_transform(df['Student_Type'])

    # 5. Outlier Treatment (Capping using IQR)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Stress_Level' in numeric_cols: numeric_cols.remove('Stress_Level')
    if 'Student_Type' in numeric_cols: numeric_cols.remove('Student_Type')
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])

    # Final Safety Net: Drop any rows that still somehow contain NaNs
    df = df.dropna()
    print("Data cleaning completed.")

    # -------------------------------------------------------------------
    # B & C. FEATURE ENGINEERING & DATASET VERSIONS
    # -------------------------------------------------------------------
    datasets = {}

    # Version 1: Cleaned Base Data (No scaling, all original features)
    X_v1 = df.drop('Stress_Level', axis=1)
    y_v1 = df['Stress_Level']
    datasets['V1_Base'] = (X_v1, y_v1, df.copy())

    # Version 2: Standardized Data (StandardScaler applied)
    scaler_std = StandardScaler()
    X_v2 = pd.DataFrame(scaler_std.fit_transform(X_v1), columns=X_v1.columns)
    datasets['V2_Standardized'] = (X_v2, y_v1.copy(), df.copy())

    # Version 3: Feature Engineered + MinMax Scaled
    df_v3 = df.copy()
    
    # Create new feature: Ratio of Study Hours to Social Media Hours
    # We use + 1 to prevent division by zero, but we also explicitly replace Inf with 0
    df_v3['Study_Social_Ratio'] = df_v3['Study_Hours'] / (df_v3['Social_Media_Hours'] + 1)
    df_v3 = df_v3.replace([np.inf, -np.inf], 0) # Safeguard against Infinity

    # Drop irrelevant feature based on understanding
    if 'Month' in df_v3.columns:
        df_v3 = df_v3.drop(['Month'], axis=1) 

    X_v3_unscaled = df_v3.drop('Stress_Level', axis=1)
    scaler_mm = MinMaxScaler()
    X_v3 = pd.DataFrame(scaler_mm.fit_transform(X_v3_unscaled), columns=X_v3_unscaled.columns)
    datasets['V3_FeatureEng_MinMax'] = (X_v3, df_v3['Stress_Level'], df_v3.copy())
    
    print("Feature engineering and dataset versioning completed.")

    # -------------------------------------------------------------------
    # D. EXPERIMENTATION (5 Models, Varying Parameters)
    # -------------------------------------------------------------------
    models = {
        'Logistic Regression': (
            LogisticRegression(max_iter=1000), 
            {'C': [0.1, 1, 10]}
        ),
        'Decision Tree': (
            DecisionTreeClassifier(random_state=42), 
            {'max_depth': [3, 5, 10, None]}
        ),
        'Random Forest': (
            RandomForestClassifier(random_state=42), 
            {'n_estimators': [50, 100], 'max_depth': [5, 10]}
        ),
        'SVM': (
            SVC(random_state=42, probability=True), 
            {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        ),
        'Gradient Boosting': (
            GradientBoostingClassifier(random_state=42), 
            {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
        )
    }

    results = []
    best_overall_model = None
    best_overall_acc = 0
    best_overall_dataset_name = ""
    best_overall_dataset_df = None

    print("Starting model training and evaluation. This may take a moment...")

    # Loop through each dataset version
    for data_name, (X, y, original_df) in datasets.items():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Loop through each model
        for model_name, (model, params) in models.items():
            grid = GridSearchCV(model, params, cv=3, scoring='accuracy', n_jobs=-1)
            grid.fit(X_train, y_train)
            
            best_model = grid.best_estimator_
            y_pred = best_model.predict(X_test)
            
            # E. PERFORMANCE COMPARISON
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            results.append({
                'Dataset Version': data_name,
                'Model': model_name,
                'Best Params': str(grid.best_params_),
                'Accuracy': round(acc, 4),
                'Precision': round(prec, 4),
                'Recall': round(rec, 4),
                'F1-Score': round(f1, 4)
            })

            # Track the absolute best model
            if acc > best_overall_acc:
                best_overall_acc = acc
                best_overall_model = best_model
                best_overall_dataset_name = data_name
                best_overall_dataset_df = original_df

    results_df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
    print("\n--- MODEL PERFORMANCE COMPARISON ---")
    print(results_df.to_string(index=False))

    # -------------------------------------------------------------------
    # F. EXPORT BEST MODEL AND DATASET FOR DJANGO
    # -------------------------------------------------------------------
    print(f"\nExporting the Best Model: {type(best_overall_model).__name__} trained on {best_overall_dataset_name}")
    
    # Save the model
    joblib.dump(best_overall_model, 'best_stress_model.pkl')
    # Save the corresponding cleaned dataset
    best_overall_dataset_df.to_csv('best_cleaned_dataset.csv', index=False)
    
    print("SUCCESS: 'best_stress_model.pkl' and 'best_cleaned_dataset.csv' saved successfully!")

if __name__ == "__main__":
    main()