import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

regressor = load_model()['model']

st.title('House Price Prediction')
st.write("""### Enter the following details to predict the price of the house""")

# Input features : 1. Bedrooms 2. Full Bath 3. Half Bath 4. Above Ground Living Area 5. Lot Area 6. Quality 7. Condition 8. Year Built 9. Year Sold

bedrooms = st.number_input("Bedrooms", 1, 10, 1)
full_bath = st.number_input("Full Bath", 1, 10, 1)
half_bath = st.number_input("Half Bath", 0, 10, 1)
above_ground_living_area = st.number_input("Above Ground Living Area", 200, 6000, 1000, 100)
lot_area = st.number_input("Lot Area", 1000, 100000, 10000, 1000)
quality = st.slider("Quality", 1, 10, 5)
condition = st.slider("Condition", 1, 10, 5)

ok = st.button("Predict")

if ok:
    input = np.array([[bedrooms, full_bath, half_bath, above_ground_living_area, lot_area, quality, condition, 2006, 2010]]).astype(np.float64)
    prediction = regressor.predict(input)
    st.write(f"""### The predicted price of the house is ${prediction[0]}""")