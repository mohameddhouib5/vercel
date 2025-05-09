import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("RandomForest_Model.pkl")

# Columns dropped during training (including 'Sold_Price')
cols_to_drop = ['Car_Id', 'SerialNumber', 'Unnamed: 16', 'Transmission.1',
                'Condition', 'Color', 'Condition_Score', 'Car_Age', 'Entertainment_Features', 'Sold_Price']

# Categorical columns used in training
categorical_cols = ['Brand', 'Model', 'Type', 'FuelType', 'Safety_Features',
                    'Region', 'Transmission', 'Accident_History']

# Load dummy frame for alignment
@st.cache_data
def load_dummy_frame():
    df = pd.read_csv("Estimacar.csv", delimiter=';')
    df.drop(columns=cols_to_drop, inplace=True)
    dummy = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return dummy.drop(columns="Price_Category", errors="ignore")

dummy_columns = load_dummy_frame().columns

# Streamlit UI
st.title("🚗 Car Price Category Estimator")

# Dropdown options
brands = ['Renault', 'Citroën', 'Peugeot', 'Ford', 'Mercedes', 'BMW', 'Toyota', 'Volkswagen', 'Hyundai', 'Kia']
types = ['SUV', 'Hatchback', 'Pickup', 'Sedan', 'Coupe']
fuel_types = ['Petrol', 'Electric', 'Hybrid', 'Diesel']
regions = ['Bizerte', 'Sousse', 'Gabès', 'Tataouine', 'Monastir', 'Kairouan', 'Tozeur', 'Sfax', 'Nabeul', 'Tunis']
transmissions = ['Manual', 'Automatic']
safety_options = ['ABS', 'Airbags', 'ESP', 'Lane Assist', 'Traction Control', 'Blind Spot Monitor']
accident_history_options = ['Yes', 'No']
models = ['Megane', 'C4', 'Partner', 'Kuga', 'C-Class', '7 Series', 'E-Class', 'Clio', '2008', 'C3', 'Hilux',
          'Golf', 'Berlingo', 'Corolla', 'Accent', 'Rio', 'GLC', 'X3', 'X1', '3008', '3 Series', 'Picanto',
          'Passat', 'Tucson', 'A-Class', 'Transit', 'Camry', 'Sprinter', 'Kadjar', '208', 'Ranger', 'Cerato',
          '5 Series', 'Rav4', 'Sportage', 'Symbol', 'C5', 'i10', 'Elantra', 'Jumpy', 'Jetta', 'Sorento', '308',
          'Focus', 'Polo', 'Yaris', 'Fiesta', 'i20', 'Tiguan', 'Captur']

# User inputs
Brand = st.selectbox("Brand", brands)
Model = st.selectbox("Model", models)
Driven_KM = st.number_input("Mileage (KM)", min_value=0)
Type = st.selectbox("Type", types)
EngineV = st.number_input("Engine Displacement (L)", min_value=0.0, step=0.1)
FuelType = st.selectbox("Fuel Type", fuel_types)
Safety_Features = st.multiselect("Safety Features", safety_options)
Region = st.selectbox("Region", regions)
Year = st.number_input("Manufacturing Year", min_value=1980, max_value=2025, value=2015)
Transmission = st.selectbox("Transmission", transmissions)
Accident_History = st.selectbox("Accident History", accident_history_options)

# Format safety features as a string
Safety_Str = ", ".join(Safety_Features)

# Create raw input DataFrame
raw_input = pd.DataFrame([{
    'Brand': Brand,
    'Model': Model,
    'Driven_KM': Driven_KM,
    'Type': Type,
    'EngineV': EngineV,
    'FuelType': FuelType,
    'Safety_Features': Safety_Str,
    'Region': Region,
    'Year': Year,
    'Transmission': Transmission,
    'Accident_History': Accident_History
}])

# OneHot encoding (same as during training)
input_encoded = pd.get_dummies(raw_input, columns=categorical_cols, drop_first=True)

# Align input with training data
input_aligned = pd.DataFrame(columns=dummy_columns)
input_aligned = pd.concat([input_aligned, input_encoded], ignore_index=True).fillna(0)
input_aligned = input_aligned[dummy_columns]

# Prediction
if st.button("🔍 Predict"):
    prediction = model.predict(input_aligned)
    result = prediction[0]

    # Price range output
    if result.lower() == "low":
        price_info = "💰 Less than 30,000 TND"
    elif result.lower() == "medium":
        price_info = "💰 Between 30,000 TND and 50,000 TND"
    else:
        price_info = "💰 Over 50,000 TND"

    st.success(f"✅ Estimated Sale Amount: **{result.capitalize()}**")
    st.info(price_info)

