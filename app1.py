import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import os
from scipy.stats import kurtosis, skew

# ==========================================
# 1. SIDEBAR MENU
# ==========================================
st.sidebar.title("🚆 Train Condition Monitoring")
app_mode = st.sidebar.radio("Select Subsystem:", ["ACV Fault Localisation", "Rail Corrugation Detection"])

# ==========================================
# 2. ACV FAULT LOCALISATION LOGIC
# ==========================================
if app_mode == "ACV Fault Localisation":
    st.title("❄️ ACV Fault Localisation")
    st.write("Upload a train telemetry file (`.xlsx`) to predict which car has a refrigerant leak.")

    @st.cache_resource
    def load_acv_model():
        return joblib.load('acv_model.pkl'), joblib.load('acv_features.pkl')
    
    # Rename the function so it doesn't clash!
    def extract_acv_features(df, cars=['01', '02', '03', '04', '05', '06', '07', '08']):
        features = []
        for car in cars:
            car_cols = [col for col in df.columns if f"Car {car}" in col]
            if not car_cols: continue
                
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

    try:
        rf_model, expected_features = load_acv_model()
        
        uploaded_file = st.file_uploader("Upload ACV Test Case (.xlsx)", type=["xlsx"])
        if uploaded_file is not None:
            st.info(f"Processing `{uploaded_file.name}`...")
            df = pd.read_excel(uploaded_file)
            test_features_df = extract_acv_features(df) # Use renamed function
            
            if not test_features_df.empty:
                for col in expected_features:
                    if col not in test_features_df.columns:
                        test_features_df[col] = 0 
                        
                X_test = test_features_df[expected_features].fillna(0)
                probs = rf_model.predict_proba(X_test)[:, 1]
                test_features_df['fault_probability'] = probs
                
                ranked_cars_df = test_features_df.sort_values(by='fault_probability', ascending=False)
                ranked_cars_list = ranked_cars_df['car_id'].tolist()
                ranked_cars_str = "|".join(ranked_cars_list)
                
                st.success("✅ ACV Prediction Complete!")
                st.write(f"**Most likely faulty car:** {ranked_cars_list[0]}")
                st.write(f"**Full ranking:** `{ranked_cars_str}`")
                
                submission_df = pd.DataFrame({'file_id': [uploaded_file.name], 'ranked_cars': [ranked_cars_str]})
                csv_buffer = io.StringIO()
                submission_df.to_csv(csv_buffer, index=False)
                
                st.download_button(label="📥 Download acv_predictions.csv", data=csv_buffer.getvalue(), file_name="acv_predictions.csv", mime="text/csv")
            else:
                st.error("Could not extract car features.")
    except Exception as e:
        st.error(f"ACV Error: {e}")

# ==========================================
# 3. RAIL CORRUGATION LOGIC
# ==========================================
elif app_mode == "Rail Corrugation Detection":
    st.title("🛤️ Rail Corrugation Detection")
    st.write("Upload an Axle-Box Vibration file (`.csv`) to detect corrugation.")
    
    @st.cache_resource
    def load_rail_model():
        return joblib.load('rail_model.pkl')
    
    # Rename function so it doesn't clash!
    def extract_rail_features(df):
        signal_cols = [c for c in df.columns if c.lower() not in ['time', 'timestamp', 'index', 'id']]
        if not signal_cols:
            signal_cols = df.columns.tolist()

        feats = {}
        for col in signal_cols:
            vals = pd.to_numeric(df[col], errors='coerce').dropna().values
            if len(vals) == 0: continue

            feats[f"{col}_mean"] = np.mean(vals)
            feats[f"{col}_std"] = np.std(vals)
            feats[f"{col}_rms"] = np.sqrt(np.mean(vals**2))
            feats[f"{col}_p2p"] = np.ptp(vals)
            feats[f"{col}_kurtosis"] = float(kurtosis(vals))
            feats[f"{col}_skew"] = float(skew(vals))

            fft_vals = np.abs(np.fft.rfft(vals))
            feats[f"{col}_fft_mean"] = np.mean(fft_vals)
            feats[f"{col}_fft_energy"] = np.sum(fft_vals**2) / len(fft_vals)
        return pd.DataFrame([feats])

    try:
        rail_model = load_rail_model()
        inv_label_map = {0: 'Normal', 1: 'Side I', 2: 'Side II'}
        
        uploaded_file = st.file_uploader("Upload Rail Data (.csv)", type=["csv"])
        if uploaded_file is not None:
            st.info("Analyzing vibration signatures...")
            df = pd.read_csv(uploaded_file)
            features_df = extract_rail_features(df) # Use renamed function
            
            X_input = features_df.fillna(0).values
            pred = rail_model.predict(X_input)[0]
            result = inv_label_map[pred]
            
            if result == 'Normal':
                st.success(f"✅ {result} - Track is healthy.")
            else:
                st.error(f"⚠️ {result} - Corrugation detected! Maintenance required.")
                
            # Create download for Rail predictions
            submission_df = pd.DataFrame({'file_id': [uploaded_file.name], 'prediction': [result]})
            csv_buffer = io.StringIO()
            submission_df.to_csv(csv_buffer, index=False)
            st.download_button(label="📥 Download rail_predictions.csv", data=csv_buffer.getvalue(), file_name="rail_predictions.csv", mime="text/csv")
            
    except Exception as e:
        st.error(f"Rail Error: {e}")