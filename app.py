# app.py
import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
import plotly.express as px
import os

# --- Set MLflow Tracking URI ---
# Ensure MLflow looks in the correct directory where your model was logged.
current_dir = os.path.dirname(os.path.abspath(__file__))
mlruns_path = os.path.join(current_dir, "mlruns")
tracking_uri = f"file:///{mlruns_path.replace(os.sep, '/')}"
mlflow.set_tracking_uri(tracking_uri)
print(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")  # Debug print
# --- End MLflow Setup ---

# Set page config
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide"
)

# Title and description
st.title("🏥 Medical Insurance Cost Predictor")
st.markdown("""
This app predicts medical insurance costs based on user input. 
It uses a machine learning model trained on a dataset of 2772 individuals.
""")

# Sidebar for user input
st.sidebar.header("👤 User Input Features")

# Function to get user input features
def user_input_features():
    age = st.sidebar.slider('Age', 18, 65, 30)
    sex = st.sidebar.selectbox('Sex', ('male', 'female'))
    bmi = st.sidebar.slider('BMI', 15.0, 55.0, 25.0)
    children = st.sidebar.slider('Number of Children', 0, 5, 0)
    smoker = st.sidebar.selectbox('Smoker', ('yes', 'no'))
    region = st.sidebar.selectbox('Region', ('northeast', 'northwest', 'southeast', 'southwest'))
    
    data = {
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
df_input = user_input_features()

# Display user input
st.subheader("User Input Parameters")
st.write(df_input)

# --- Model Loading Section ---
# Use the NEW Run ID from your re-run notebook cell
best_rf_run_id = "4b16c04cc334458fb51a0e2f44ee7b04" # <--- UPDATE THIS WITH YOUR ACTUAL NEW RUN ID

model = None
model_load_error = ""

# --- Strategy 1: Try loading using the specific Run ID ---
try:
    st.info(f"Attempting to load model from Run ID: `{best_rf_run_id}`")
    model = mlflow.pyfunc.load_model(model_uri=f"runs:/{best_rf_run_id}/model")
    if model is not None:
        st.success("✅ Model loaded successfully using Run ID!")
    else:
        model_load_error = "Model loading function returned None (Run ID)."
        st.warning(model_load_error)
except Exception as e1:
    model_load_error = f"Failed to load model using Run ID: {e1}"
    st.warning(model_load_error)

# --- Strategy 2: Fallback - Try loading directly from artifact path ---
if model is None:
    try:
        # Construct the path explicitly for Windows compatibility
        model_artifact_uri = os.path.join(mlruns_path, "0", best_rf_run_id, "artifacts", "model")
        # Convert path separators for URI
        model_artifact_uri = model_artifact_uri.replace(os.sep, '/')
        model_uri = f"file:///{model_artifact_uri}"
        
        st.info(f"Attempting to load model from local path URI: `{model_uri}`")
        model = mlflow.pyfunc.load_model(model_uri=model_uri)
        
        if model is not None:
            st.success("✅ Model loaded successfully from local artifact path!")
        else:
            model_load_error = "Model loading function returned None (local path)."
            st.warning(model_load_error)
    except Exception as e2:
        model_load_error = f"Failed to load model from local path: {e2}"
        st.warning(model_load_error)

# --- Final Check before proceeding ---
if model is None:
    st.error(f"❌ Could not load any model. Last error: {model_load_error}")
    st.stop()
else:
    st.write(f"Model type: {type(model)}") # Debug info
# --- End Model Loading Section ---

# --- Prepare Input for Prediction (Fixed Data Types) ---
# One-hot encode categorical variables to match training data format
input_encoded = pd.get_dummies(df_input, columns=['sex', 'smoker', 'region'], drop_first=True)

# Ensure all columns from training are present (in case some dummy variables are missing)
expected_columns = [
    'age', 'bmi', 'children', 
    'sex_male', 'smoker_yes', 
    'region_northwest', 'region_southeast', 'region_southwest'
]

# Add missing columns with default value 0 (integer)
for col in expected_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# --- CRITICAL FIX: Ensure boolean columns are boolean type ---
# This prevents the schema enforcement error from MLflow
boolean_columns = ['sex_male', 'smoker_yes', 'region_northwest', 'region_southeast', 'region_southwest']
for col in boolean_columns:
    if col in input_encoded.columns:
        # Safely convert to boolean. Integers 0/1 will become False/True
        input_encoded[col] = input_encoded[col].astype(bool)

# Reorder columns to match the exact order expected by the model
input_encoded = input_encoded.reindex(columns=expected_columns, fill_value=0)

# Create interaction features (matching your training preprocessing)
input_encoded['age_smoker'] = input_encoded['age'] * input_encoded['smoker_yes']
input_encoded['bmi_smoker'] = input_encoded['bmi'] * input_encoded['smoker_yes']
input_encoded['age_bmi'] = input_encoded['age'] * input_encoded['bmi']

# --- Optional Debug: Show final input schema ---
# st.write("Debug: Final Input DataFrame Schema:")
# st.write(input_encoded.dtypes)
# st.write("Debug: Final Input DataFrame:")
# st.write(input_encoded)

# --- Make Prediction ---
try:
    prediction = model.predict(input_encoded)
    st.subheader("Predicted Insurance Cost")
    st.success(f"${prediction[0]:,.2f}")
except Exception as e:
    st.error(f"Error making prediction: {str(e)}")
    st.stop()

# --- Load the original dataset for visualizations ---
@st.cache_data
def load_data():
    # Make sure this path is correct relative to where you run 'streamlit run app.py'
    data_path = os.path.join(current_dir, "data", "raw", "medical_insurance.csv")
    data = pd.read_csv(data_path)
    return data

df = load_data()

# EDA Visualizations
st.subheader("📊 Exploratory Data Analysis Insights")

# Distribution of charges
fig_charges = px.histogram(df, x="charges", nbins=50, 
                           title="Distribution of Insurance Charges",
                           labels={"charges": "Charges ($)"})
fig_charges.update_layout(showlegend=False)
st.plotly_chart(fig_charges, use_container_width=True)

# Charges by smoking status
fig_smoker = px.box(df, x="smoker", y="charges", 
                    title="Insurance Charges by Smoking Status",
                    labels={"smoker": "Smoker", "charges": "Charges ($)"})
st.plotly_chart(fig_smoker, use_container_width=True)

# Charges by age
fig_age = px.scatter(df, x="age", y="charges", color="smoker",
                     title="Insurance Charges by Age and Smoking Status",
                     labels={"age": "Age", "charges": "Charges ($)", "smoker": "Smoker"})
st.plotly_chart(fig_age, use_container_width=True)

# Charges by BMI
fig_bmi = px.scatter(df, x="bmi", y="charges", color="smoker",
                     title="Insurance Charges by BMI and Smoking Status",
                     labels={"bmi": "BMI", "charges": "Charges ($)", "smoker": "Smoker"})
st.plotly_chart(fig_bmi, use_container_width=True)

# Average charges by region
avg_charges_region = df.groupby("region")["charges"].mean().reset_index()
fig_region = px.bar(avg_charges_region, x="region", y="charges",
                    title="Average Insurance Charges by Region",
                    labels={"region": "Region", "charges": "Average Charges ($)"})
st.plotly_chart(fig_region, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit and MLflow")

