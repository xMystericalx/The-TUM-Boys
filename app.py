import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import os

# --- 1. Load the pre-trained model and features ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('acv_model.pkl')
        features = joblib.load('acv_features.pkl')
        return model, features
    except Exception as e:
        st.error(f"Error loading model files: {e}. Please ensure 'acv_model.pkl' and 'acv_features.pkl' are in the same folder.")
        return None, None

rf_model, expected_features = load_model()

# --- 2. Feature Extraction Function (Same as training) ---
def extract_features(df, cars=['01', '02', '03', '04', '05', '06', '07', '08']):
    features = []
    for car in cars:
        car_cols = [col for col in df.columns if f"Car {car}" in col]
        if not car_cols:
            continue
            
        car_df = df[car_cols]
        numeric_cols = car_df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            mean_vals = car_df[numeric_cols].mean()
            std_vals = car_df[numeric_cols].std()
            max_vals = car_df[numeric_cols].max()
            min_vals = car_df[numeric_cols].min()
            
            indoor_col = next((col for col in numeric_cols if 'Indoor Average Temperature' in col), None)
            control_col = next((col for col in numeric_cols if 'Control Temperature (Cooling)' in col), None)
            
            diff_mean = 0
            if indoor_col and control_col:
                diff_mean = (car_df[indoor_col] - car_df[control_col]).mean()
                
            car_features = pd.concat([
                mean_vals.add_suffix('_mean'), 
                std_vals.add_suffix('_std'),
                max_vals.add_suffix('_max'),
                min_vals.add_suffix('_min')
            ])
            
            standardized_features = {}
            for index, value in car_features.items():
                clean_name = index.split(" - ", 1)[1] if " - " in index else index
                standardized_features[clean_name] = value
                
            standardized_features['Temp_Diff_mean'] = diff_mean
            
            feat_series = pd.Series(standardized_features)
            feat_series['car_id'] = car
            features.append(feat_series)
            
    return pd.DataFrame(features)

# --- 3. Streamlit UI ---
st.title("🚆 Train ACV Fault Localisation")
st.write("Upload a train telemetry file (`.xlsx`) to predict which car has a refrigerant leak.")

uploaded_file = st.file_uploader("Upload ACV Test Case (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    st.info(f"Processing `{uploaded_file.name}`...")
    
    # Read the excel file
    try:
        df = pd.read_excel(uploaded_file)
        
        # Extract features
        test_features_df = extract_features(df)
        
        if not test_features_df.empty:
            # Align features with the training set
            for col in expected_features:
                if col not in test_features_df.columns:
                    test_features_df[col] = 0  # Fill missing columns with 0
                    
            X_test = test_features_df[expected_features]
            X_test = X_test.fillna(0)
            
            # Predict probabilities
            probs = rf_model.predict_proba(X_test)[:, 1]
            test_features_df['fault_probability'] = probs
            
            # Rank cars
            ranked_cars_df = test_features_df.sort_values(by='fault_probability', ascending=False)
            ranked_cars_list = ranked_cars_df['car_id'].tolist()
            ranked_cars_str = "|".join(ranked_cars_list)
            
            st.success("✅ Prediction Complete!")
            st.subheader("Results:")
            st.write(f"**Most likely faulty car:** {ranked_cars_list[0]}")
            st.write(f"**Full ranking (Most to Least likely):** `{ranked_cars_str}`")
            
            # Prepare CSV for download
            submission_df = pd.DataFrame({
                'file_id': [uploaded_file.name],
                'ranked_cars': [ranked_cars_str]
            })
            
            # Convert DF to CSV in memory
            csv_buffer = io.StringIO()
            submission_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📥 Download acv_predictions.csv",
                data=csv_data,
                file_name="acv_predictions.csv",
                mime="text/csv"
            )
            
            # Show probability breakdown for transparency
            with st.expander("View Probability Breakdown"):
                st.dataframe(ranked_cars_df[['car_id', 'fault_probability']].reset_index(drop=True))
                
        else:
            st.error("Could not extract car features from this file. Please check the file format.")
            
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")