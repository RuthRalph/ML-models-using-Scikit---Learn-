import streamlit as st
import pandas as pd
import joblib
import numpy as np
import random
from datetime import datetime, timedelta

st.set_page_config(page_title="Predictive Maintenance", layout="wide")
st.title("Demo Predictive Maintenance Dashboard")
st.write(".")

# Load your ACTUAL trained model
try:
    model = joblib.load('predictive_maintenance_model.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    st.sidebar.success("Adjust to Load Prediction")
    st.sidebar.info(f" {len(feature_columns)} features loaded")
except Exception as e:
    st.sidebar.error(f"❌ Model loading failed: {e}")
    model = None
    feature_columns = []

# Sidebar inputs
st.sidebar.header("🔧 Machine Parameters")
air_temperature = st.sidebar.slider("Air Temperature [K]", 295.0, 310.0, 298.2)
process_temperature = st.sidebar.slider("Process Temperature [K]", 305.0, 315.0, 308.6)
rotational_speed = st.sidebar.slider("Rotational Speed [rpm]", 1100, 3000, 1500)
torque = st.sidebar.slider("Torque [Nm]", 30.0, 80.0, 40.0)
tool_wear = st.sidebar.slider("Tool Wear [min]", 0, 300, 0)
machine_type = st.sidebar.selectbox("Machine Type", ["Low", "Medium", "High"])

# Initialize session state for dynamic data
if 'equipment_data' not in st.session_state:
    st.session_state.equipment_data = None
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = "Never"

# Function to generate realistic equipment data
def generate_live_equipment_data():
    machines = []
    base_time = datetime.now()
    
    for i in range(6):  # 6 machines
        machine_id = f"M{100 + i}"
        
        # Realistic variations
        temp_variation = random.uniform(-2, 4)
        speed_variation = random.uniform(-80, 120)
        wear_variation = random.randint(10, 180)
        
        # Machine type distribution
        status_types = ["Low", "Medium", "High"]
        machine_status = random.choice(status_types)
        
        # Generate realistic readings
        base_temp = 298 + temp_variation
        base_speed = 1500 + speed_variation
        base_torque = 40 + random.uniform(-4, 6)
        
        # Create input for prediction
        input_data = {
            'Air temperature [K]': [base_temp],
            'Process temperature [K]': [base_temp + 9 + random.uniform(-1, 2)],
            'Rotational speed [rpm]': [base_speed],
            'Torque [Nm]': [base_torque],
            'Tool wear [min]': [wear_variation],
            'Type_L': [1 if machine_status == "Low" else 0],
            'Type_M': [1 if machine_status == "Medium" else 0],
            'Type_H': [1 if machine_status == "High" else 0]
        }
        
        # Calculate health score
        health_score = 85 - (wear_variation / 4) + random.uniform(-5, 5)
        health_score = max(10, min(98, health_score))
        
        # Determine status
        if health_score > 80:
            status_emoji = "✅"
            status_text = "Excellent"
        elif health_score > 60:
            status_emoji = "⚠️"
            status_text = "Good"
        elif health_score > 40:
            status_emoji = "🔶"
            status_text = "Warning"
        else:
            status_emoji = "🚨"
            status_text = "Critical"
        
        days_since_maintenance = random.randint(1, 60)
        
        machines.append({
            'Machine ID': machine_id,
            'Type': machine_status,
            'Temp [K]': round(base_temp, 1),
            'Speed [rpm]': int(base_speed),
            'Health Score': round(health_score, 1),
            'Status': f"{status_emoji} {status_text}",
            'Last Maintenance': days_since_maintenance
        })
    
    return pd.DataFrame(machines)

# Main app logic - EVERYTHING responds to button clicks
st.sidebar.header("Actions")

# Single machine prediction
if st.sidebar.button("Analyze Single Machine"):
    st.header("Single Machine Analysis")
    
    if model is not None and len(feature_columns) > 0:
        # Create input data
        input_data = {
            'Air temperature [K]': [air_temperature],
            'Process temperature [K]': [process_temperature],
            'Rotational speed [rpm]': [rotational_speed],
            'Torque [Nm]': [torque],
            'Tool wear [min]': [tool_wear],
            'Type_L': [1 if machine_type == "Low" else 0],
            'Type_M': [1 if machine_type == "Medium" else 0],
            'Type_H': [1 if machine_type == "High" else 0]
        }
        
        input_df = pd.DataFrame(input_data)
        
        # Ensure columns match
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_columns]
        
        # Get prediction
        failure_probability = model.predict_proba(input_df)[0][1]
        health_score = (1 - failure_probability) * 100
        
        st.success("**PREDICTIONS**")
        
    else:
        # Demo mode
        health_score = 100 - (air_temperature - 298) * 2 - (tool_wear / 10)
        health_score = max(0, min(100, health_score))
        failure_probability = (100 - health_score) / 100
        st.warning("⚠️ **DEMO MODE** - Using simulated predictions")

    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Health Score", f"{health_score:.1f}%")
    
    with col2:
        st.metric("Failure Probability", f"{failure_probability:.1%}")
    
    with col3:
        if failure_probability < 0.2:
            st.success("✅ EXCELLENT")
        elif failure_probability < 0.5:
            st.warning("⚠️ WARNING")
        else:
            st.error("🚨 CRITICAL")
    
    # Recommendations
    if failure_probability < 0.2:
        st.info("🎯 **Recommendation**: Continue normal operations")
        st.balloons()
    elif failure_probability < 0.5:
        st.warning("🎯 **Recommendation**: Monitor closely - schedule check-up")
    else:
        st.error("🎯 **Recommendation**: Schedule maintenance immediately!")

# Factory monitoring
if st.sidebar.button("View Factory Status"):
    st.header("Live Factory Monitoring")
    
    # Generate new equipment data
    st.session_state.equipment_data = generate_live_equipment_data()
    st.session_state.last_refresh = datetime.now().strftime("%H:%M:%S")
    
    st.info(f"🔄 Last updated: {st.session_state.last_refresh}")
    
    if st.session_state.equipment_data is not None:
        # Display equipment data
        st.dataframe(
            st.session_state.equipment_data,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Health Score": st.column_config.ProgressColumn(
                    "Health Score",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                )
            }
        )
        
        # Summary statistics
        st.subheader("📈 Factory Health Summary")
        summary_cols = st.columns(3)
        
        with summary_cols[0]:
            excellent = len(st.session_state.equipment_data[st.session_state.equipment_data['Health Score'] > 80])
            st.metric("Excellent", f"{excellent}/6")
        
        with summary_cols[1]:
            warning = len(st.session_state.equipment_data[st.session_state.equipment_data['Health Score'] <= 60])
            st.metric("Needs Attention", f"{warning}/6")
        
        with summary_cols[2]:
            avg_health = st.session_state.equipment_data['Health Score'].mean()
            st.metric("Avg Health", f"{avg_health:.1f}%")

# Business impact (always visible but interactive)
# st.header("💰 Business Impact")

# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("❌ Without AI System")
#     st.write("• 38% failure detection")
#     st.write("• $40,000 emergency repairs")
#     st.write("• 8+ hours unplanned downtime")
#     st.write("• $2.4M annual losses")

# with col2:
#     st.subheader("✅ With AI System")
#     st.write("• **72.1% failure detection**")
#     st.write("• $1,000 planned maintenance")
#     st.write("• 2 hours planned downtime") 
#     st.write("• **$1.3M annual savings**")

# Interactive savings calculator
st.subheader("Custom Savings Calculator")

emergency_cost = st.number_input("Emergency repair cost ($)", 1000, 100000, 40000, 1000)
planned_cost = st.number_input("Planned maintenance cost ($)", 100, 10000, 1000, 100)
failures_per_year = st.slider("Failures per year", 1, 50, 10)

savings_per_failure = emergency_cost - planned_cost
annual_savings = savings_per_failure * failures_per_year * 0.721  # 72.1% detection

st.success(f"**Potential annual savings: ${annual_savings:,.0f}**")

st.markdown("---")
st.success(".")

# Instructions
st.sidebar.markdown("---")
st.sidebar.info("""
**How to use:**
1. **Analyze Single Machine** - Test parameters
2. **View Factory Status** - See all equipment
3. Adjust sliders for different scenarios
""")