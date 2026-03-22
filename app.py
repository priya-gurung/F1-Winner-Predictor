import streamlit as st
import pandas as pd
import fastf1
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# --- 1. SETUP & CACHING ---
st.set_page_config(page_title="2026 F1 Predictor", page_icon="🏎️")

if not os.path.exists('f1_cache'):
    os.makedirs('f1_cache')
fastf1.Cache.enable_cache('f1_cache')

# --- 2. DATA LOADING & TRAINING (Runs once on startup) ---
@st.cache_resource
def train_f1_model():
    all_race_data = []
    # Loading 2024, 2025, and the start of 2026
    for y in [2024, 2025, 2026]:
        try:
            # Using Australia as the baseline for the weekend build
            session = fastf1.get_session(y, 'Australia', 'R')
            session.load()
            df = session.results.copy()
            df['Year'] = y
            all_race_data.append(df)
        except Exception as e:
            st.error(f"Error loading {y}: {e}")
    
    final_df = pd.concat(all_race_data)
    
    # Pre-processing
    model_df = final_df[final_df['Status'] == 'Finished'].copy()
    model_df['Weight'] = model_df['Year'].apply(lambda x: 1.0 if x == 2026 else 0.3)
    
    d_enc = LabelEncoder()
    t_enc = LabelEncoder()
    model_df['Driver_ID'] = d_enc.fit_transform(model_df['Abbreviation'])
    model_df['Team_ID'] = t_enc.fit_transform(model_df['TeamName'])
    
    # Train Model
    features = ['GridPosition', 'Team_ID', 'Driver_ID']
    X = model_df[features]
    y = model_df['ClassifiedPosition']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y, sample_weight=model_df['Weight'])
    
    return model, d_enc, t_enc

with st.status("🏎️ Initializing 2026 Predictor...", expanded=True) as status:
    st.write("Downloading 2024 season data...")
    # (The model training happens here)
    model, driver_encoder, team_encoder = train_f1_model()
    status.update(label="✅ Data Loaded! Ready for Suzuka.", state="complete", expanded=False)

# --- 3. STREAMLIT UI ---
st.title("🏁 F1 Winner Predictor: 2026 Era")
st.markdown(f"Current Date: **March 22, 2026** | Next Race: **Japanese GP**")

with st.sidebar:
    st.header("Race Setup")
    gp = st.selectbox("Select Grand Prix", ["Japanese GP", "Chinese GP", "Australian GP"])
    st.info("Mercedes and Ferrari are currently leading the 2026 standings.")

# Input Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        driver = st.selectbox("Select Driver", driver_encoder.classes_)
        grid_pos = st.number_input("Starting Grid Position", min_value=1, max_value=20, value=1)
    with col2:
        team = st.selectbox("Select Team", team_encoder.classes_)
    
    predict_btn = st.form_submit_button("Calculate Predicted Finish")

if predict_btn:
    d_id = driver_encoder.transform([driver])[0]
    t_id = team_encoder.transform([team])[0]
    
    prediction = model.predict([[grid_pos, t_id, d_id]])[0]
    
    st.subheader(f"Results for {driver}")
    st.metric(label="AI Predicted Finishing Position", value=f"P{prediction:.1f}")
    
    if prediction < grid_pos:
        st.success(f"📈 Predicted to gain {grid_pos - prediction:.1f} spots!")
    else:
        st.warning(f"📉 Predicted to drop {prediction - grid_pos:.1f} spots.")