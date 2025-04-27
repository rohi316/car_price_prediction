import numpy as np
import pandas as pd
import pickle as pkl
import streamlit as st

rf = pkl.load(open('rf.pkl','rb'))


st.header('Car Price prediction Random Forest Regression Model')

car_data = pd.read_csv('car data.csv')

brand = lambda x: x.split(' ')[0]
car_data['Car_Name'] = car_data['Car_Name'].apply(brand)

car_name = st.selectbox('Select Car Brand',car_data['Car_Name'].unique())

year = st.slider('Manufatured Year of Car',2007,2014)

kms_driven = st.slider('No. of kms Driven',500,500000)

fuel_type = st.selectbox('Select Fuel type',car_data['Fuel_Type'].unique())

seller_type = st.selectbox('Select Seller type',car_data['Seller_Type'].unique())

transmission = st.selectbox('Select Transsmission',car_data['Transmission'].unique())

owner = st.selectbox('Select Owner',car_data['Owner'].unique())

present_price = st.slider('Present price of Car',30000,1000000)

if st.button("Predict"):

    # x_test - dataframe with test data values of 1 row 
    x_test = pd.DataFrame(
        [[present_price, kms_driven, owner, year, car_name, fuel_type, seller_type, transmission]],
        columns=['Present_Price', 'Kms_Driven', 'Owner', 'Year', 'Car_Name_encoded', 'Fuel_Type_encoded', 'Seller_Type_encoded', 'Transmission_encoded']
    )

    x_test['Fuel_Type_encoded'].replace(['Petrol', 'Diesel', 'CNG'],[2, 1, 0], inplace=True)
    x_test['Seller_Type_encoded'].replace(['Dealer', 'Individual'],[0,1], inplace=True)
    x_test['Transmission_encoded'].replace(['Manual', 'Automatic'],[1,0], inplace=True)
    x_test['Car_Name_encoded'].replace(['ritz', 'sx4', 'ciaz', 'wagon', 'swift', 'vitara', 's', 'alto',
       'ertiga', 'dzire', 'ignis', '800', 'baleno', 'omni', 'fortuner',
       'innova', 'corolla', 'etios', 'camry', 'land', 'Royal', 'UM',
       'KTM', 'Bajaj', 'Hyosung', 'Mahindra', 'Honda', 'Yamaha', 'TVS',
       'Hero', 'Activa', 'Suzuki', 'i20', 'grand', 'i10', 'eon', 'xcent',
       'elantra', 'creta', 'verna', 'city', 'brio', 'amaze', 'jazz'],
                          [36, 39, 18, 42, 38, 41, 37, 13, 25, 22, 31,  0, 15, 35, 27, 32, 20,
       26, 17, 34,  8, 11,  6,  2,  5,  7,  4, 12, 10,  3,  1,  9, 30, 28,
       29, 24, 43, 23, 21, 40, 19, 16, 14, 33]
                          ,inplace=True)
    x_test['Present_Price'] = x_test['Present_Price']*(10**(-5))

    selling_price_predicted = rf.predict(x_test)

    st.markdown('<p style="font-size: 20px;">Predicted Selling Car Price is around</p> <p style="color: skyblue ;font-size: 20px;"> ₹ '+ str(round(selling_price_predicted[0][0]*(10**5),2))+"</p>",unsafe_allow_html=True)
