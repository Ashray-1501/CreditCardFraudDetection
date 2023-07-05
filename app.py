import streamlit as st
import pandas as pd
import prediction

# Define the DataFrame columns list
columns = ['trans_date_trans_time', 'cc_num', 'merchant', 'category',
           'amt', 'first', 'last', 'gender', 'street', 'city', 'state', 'zip',
           'lat', 'long', 'city_pop', 'job', 'dob', 'trans_num', 'unix_time',
           'merch_lat', 'merch_long']

col1, col2, col3 = st.columns(3)
 
with col1:
    trans_date_trans_time = st.text_input("trans_date_trans_time")
    category = st.text_input("category")
    last = st.text_input("last")
    city = st.text_input("city")
    lat = st.text_input("lat")
    job = st.text_input("job")
    unix_time = st.text_input("unix_time")
with col2:
    cc_num = st.text_input('cc_num')
    amt = st.text_input("amt")
    gender = st.text_input("gender")
    state = st.text_input("state")
    long = st.text_input("long")
    dob = st.text_input("dob")
    merch_lat = st.text_input("merch_lat")
with col3:
    merchant = st.text_input("merchant")
    first = st.text_input("first")
    street = st.text_input("street")
    zip = st.text_input("zip")
    city_pop = st.text_input("city_pop")
    trans_num = st.text_input("trans_num")
    merch_long = st.text_input("merch_long")


btn_predict = st.button("Predict")

if btn_predict:
    data = {
"trans_date_trans_time":[trans_date_trans_time]
,"category":[category]
,"last":[last]
,"city":[city]
,"lat":[float(lat)]
,"job":[job]
,"unix_time":[int(unix_time)]
,"cc_num":[int(cc_num)]
,"amt":[float(amt)]
,"gender":[gender]
,"state":[state]
,"long":[float(long)]
,"dob":[dob]
,"merch_lat":[float(merch_lat)]
,"merchant":[merchant]
,"first":[first]
,"street":[street]
,"zip":[int(zip)]
,"city_pop":[int(city_pop)]
,"trans_num":[trans_num]
,"merch_long":[float(merch_long)]
    }
    print(data)
    data = pd.DataFrame(data)
    st.write(data)
    result = prediction.predict(data)
    if result[0] == 0:
        st.info("No Fraud Detected")
    else:
        st.info("Fraud Detected")
    
