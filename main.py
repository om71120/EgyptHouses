###import library
from utlis import process_new
import streamlit as st
import numpy as np
import joblib

##load the model
model = joblib.load("model_RF")

def house_price():

    st.title(" Egypt house  price .....")
    st.markdown("<hr>",unsafe_allow_html=True)
    ##input filed
    Type=st.selectbox('Type',options=[ 'Apartment','Chalet','Stand Alone Villa','Town House','Twin House','Duplex','Standalone Villa','Penthouse','Twin house','Studio'])   
    Price= st.slider("Price", min_value=30000.0, max_value=13300000.0)
    Bedrooms = st.slider("Bedrooms", min_value=2.0, max_value=5.0)
    Bathrooms = st.slider("Bathrooms", min_value=1.0, max_value=7.0)
    Area=st.slider('Area',min_value=10.0, max_value=995.0)
    
    st.markdown("<hr>",unsafe_allow_html=True)

    if st.button('predict Furnished ....'):
        
        new_data=np.array([Type,Price,Bedrooms,Bathrooms,Area])
      ##call the function from ulits.py to aplly the pipline
    x_processed=process_new(x_new=new_data)
      
      ##predict using model 
    y_pred=model.predict(x_processed)

      ##Display Result 
    st.success(f'predict xG is{y_pred}')

    return None


if __name__ == "__main__":
    ##call function
    house_price()
