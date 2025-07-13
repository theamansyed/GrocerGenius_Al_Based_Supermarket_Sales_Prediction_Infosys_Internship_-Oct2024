import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

st.set_page_config(page_title="Grocery Sales Predictor", layout="centered")

# Load model from 'models' subfolder
model_path = os.path.join(os.getcwd(), "models", "xgboost_sales_model.pkl")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found at: {model_path}")

with open(model_path, "rb") as f:
    model = pickle.load(f)

# Expected feature columns
expected_columns = [
    'Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Size', 'Outlet_Location_Type',
    'Item_Fat_Content_Low_Fat', 'Item_Fat_Content_Regular', 'Outlet_Type_Grocery_Store',
    'Outlet_Type_Supermarket_Type1', 'Outlet_Type_Supermarket_Type2', 'Outlet_Type_Supermarket_Type3',
    'Item_Type_Baking_Goods', 'Item_Type_Breads', 'Item_Type_Breakfast', 'Item_Type_Canned', 'Item_Type_Dairy',
    'Item_Type_Frozen_Foods', 'Item_Type_Fruits_and_Vegetables', 'Item_Type_Hard_Drinks',
    'Item_Type_Health_and_Hygiene', 'Item_Type_Household', 'Item_Type_Meat', 'Item_Type_Others',
    'Item_Type_Seafood', 'Item_Type_Snack_Foods', 'Item_Type_Soft_Drinks', 'Item_Type_Starchy_Foods',
    'Outlet_Identifier_LOO', 'Outlet_Age', 'Visibility_Percentage', 'Price_Per_Weight',
    'Visibility_to_MRP_Ratio', 'Discount_Potential'
]

# App Title
st.markdown("""
    <h1 style='text-align: center; color: #004080;'>🛒 Grocery Sales Predictor</h1>
""", unsafe_allow_html=True)

# Two-column layout
col1, col2 = st.columns(2)

with col1:
    item_id = st.text_input("Item Identifier", value="DRC01")
    item_weight = st.number_input("Item Weight", value=9.30)
    item_visibility = st.number_input("Item Visibility", value=0.02)
    item_mrp = st.number_input("Item MRP", value=249.81)
    item_fat = st.selectbox("Item Fat Content", ["Low Fat", "Regular"])
    item_type = st.selectbox("Item Type", [
        'Baking Goods', 'Breads', 'Breakfast', 'Canned', 'Dairy', 'Frozen Foods',
        'Fruits and Vegetables', 'Hard Drinks', 'Health and Hygiene', 'Household', 'Meat',
        'Others', 'Seafood', 'Snack Foods', 'Soft Drinks', 'Starchy Foods'
    ])

with col2:
    outlet_id = st.text_input("Outlet Identifier", value="OUT049")
    outlet_size_raw = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
    outlet_location_type_raw = st.selectbox("Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"])
    outlet_type = st.selectbox("Outlet Type", ["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"])
    outlet_year = st.number_input("Outlet Establishment Year", min_value=1985, max_value=2025, value=1999)

# Manual encoding for categorical variables
outlet_size_map = {"Small": 0, "Medium": 1, "High": 2}
outlet_location_map = {"Tier 3": 0, "Tier 2": 1, "Tier 1": 2}

outlet_size = outlet_size_map[outlet_size_raw]
outlet_location_type = outlet_location_map[outlet_location_type_raw]

# Feature Engineering
store_age = datetime.now().year - outlet_year
outlet_identifier_loo = 0  # fallback

# Calculated Features
visibility_pct = item_visibility * 100
price_per_weight = item_mrp / item_weight if item_weight != 0 else 0
visibility_to_mrp = item_visibility / item_mrp if item_mrp != 0 else 0

# Base Features
features = {
    'Item_Weight': item_weight,
    'Item_Visibility': item_visibility,
    'Item_MRP': item_mrp,
    'Outlet_Size': outlet_size,
    'Outlet_Location_Type': outlet_location_type,
    'Outlet_Identifier_LOO': outlet_identifier_loo,
    'Outlet_Age': store_age,
    'Visibility_Percentage': visibility_pct,
    'Price_Per_Weight': price_per_weight,
    'Visibility_to_MRP_Ratio': visibility_to_mrp,
    'Discount_Potential': max(0, 300 - item_mrp)
}

# Manual One-Hot Encoding
fat_content_val = item_fat
outlet_type_val = outlet_type
item_type_val = item_type

for val in ["Low Fat", "Regular"]:
    features[f"Item_Fat_Content_{val.replace(' ', '_')}"] = 1 if fat_content_val == val else 0

for val in ["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"]:
    features[f"Outlet_Type_{val.replace(' ', '_')}"] = 1 if outlet_type_val == val else 0

for val in [
    'Baking Goods', 'Breads', 'Breakfast', 'Canned', 'Dairy', 'Frozen Foods',
    'Fruits and Vegetables', 'Hard Drinks', 'Health and Hygiene', 'Household', 'Meat',
    'Others', 'Seafood', 'Snack Foods', 'Soft Drinks', 'Starchy Foods']:
    features[f"Item_Type_{val.replace(' ', '_')}"] = 1 if item_type_val == val else 0

# Assemble DataFrame
input_df = pd.DataFrame([features])

# Add any missing columns
for col in expected_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder
input_df = input_df[expected_columns]

# Prediction
if st.button("Predict Sales"):
    try:
        prediction = model.predict(input_df)[0]
        st.markdown(f"""
        <div style='text-align: center; padding: 25px; border-radius: 10px; background-color: #e6f2ff; color: #003366;'>
            <h2 style='color: #004080;'>🧾 Prediction Result</h2>
            <p style='font-size: 20px;'><strong>Predicted Sales:</strong> ₹ {prediction:,.2f}</p>
            <p style='font-size: 18px;'><strong>Item Identifier:</strong> {item_id}</p>
            <p style='font-size: 18px;'><strong>Outlet Identifier:</strong> {outlet_id}</p>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")