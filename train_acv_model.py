import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier

# 1. Define files and load labels
train_files = [
    'acv_case_01.xlsx', 'acv_case_02.xlsx', 'acv_case_03.xlsx', 
    'acv_case_04.xlsx', 'acv_case_05.xlsx', 'acv_case_06.xlsx'
]
labels_df = pd.read_csv('Train_Labels.csv')
faulty_car_map = dict(zip(labels_df['filename'], labels_df['faulty_car']))

# 2. Feature Extraction Function
def extract_features(df, cars=['01', '02', '03', '04', '05', '06', '07', '08']):
    features = []
    
    for car in cars:
        # Find columns belonging to this specific car
        car_cols = [col for col in df.columns if f"Car {car}" in col]
        if not car_cols:
            continue
            
        car_df = df[car_cols]
        numeric_cols = car_df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            # Calculate summary statistics over the time series
            mean_vals = car_df[numeric_cols].mean()
            std_vals = car_df[numeric_cols].std()
            max_vals = car_df[numeric_cols].max()
            min_vals = car_df[numeric_cols].min()
            
            # Domain specific feature: difference between indoor and control temp
            indoor_col = next((col for col in numeric_cols if 'Indoor Average Temperature' in col), None)
            control_col = next((col for col in numeric_cols if 'Control Temperature (Cooling)' in col), None)
            
            diff_mean = 0
            if indoor_col and control_col:
                diff_mean = (car_df[indoor_col] - car_df[control_col]).mean()
                
            # Combine all features
            car_features = pd.concat([
                mean_vals.add_suffix('_mean'), 
                std_vals.add_suffix('_std'),
                max_vals.add_suffix('_max'),
                min_vals.add_suffix('_min')
            ])
            
            # Standardize names (remove "Car XX - " so features align across all cars)
            standardized_features = {}
            for index, value in car_features.items():
                clean_name = index.split(" - ", 1)[1] if " - " in index else index
                standardized_features[clean_name] = value
                
            standardized_features['Temp_Diff_mean'] = diff_mean
            
            feat_series = pd.Series(standardized_features)
            feat_series['car_id'] = car
            features.append(feat_series)
            
    return pd.DataFrame(features)

# 3. Process all training data
print("Extracting features from training files...")
all_train_data = []

for file in train_files:
    if os.path.exists(file):
        df = pd.read_excel(file)
        # Format the target label (e.g., 1 -> '01')
        faulty_car_id = str(faulty_car_map[file]).zfill(2) 
        
        car_features_df = extract_features(df)
        if not car_features_df.empty:
            # Label = 1 if this is the faulty car, 0 otherwise
            car_features_df['is_faulty'] = (car_features_df['car_id'] == faulty_car_id).astype(int)
            all_train_data.append(car_features_df)
    else:
        print(f"Warning: {file} not found in current directory.")

# Combine all extracted data
combined_train_df = pd.concat(all_train_data, ignore_index=True)
combined_train_df = combined_train_df.fillna(0) # Handle missing columns across different files

# 4. Prepare data for model training
# Drop non-feature columns
feature_cols = [col for col in combined_train_df.columns if col not in ['car_id', 'is_faulty']]

X = combined_train_df[feature_cols]
y = combined_train_df['is_faulty']

# 5. Train the Model
print(f"Training Random Forest with {len(feature_cols)} features...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# 6. Save the Model and Feature Columns
joblib.dump(rf_model, 'acv_model.pkl')
joblib.dump(feature_cols, 'acv_features.pkl')

print("Success! Saved 'acv_model.pkl' and 'acv_features.pkl'.")