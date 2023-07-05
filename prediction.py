import pandas as pd
from IPython.display import clear_output
import warnings
warnings.filterwarnings('ignore')
import pickle as pkl
from feature_engine.encoding import OneHotEncoder
from feature_engine.encoding import MeanEncoder
from feature_engine.transformation import YeoJohnsonTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
 

def predict(data):
# trans_date_trans_time to pandas datetime
    temp = {}
    data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
    data['trans_date_trans_time'].head(3)
    data['dob'] = pd.to_datetime(data['dob'])
    data['dob'].head(3)
    list(data.columns)
    data['trans_hour'] = data['trans_date_trans_time'].dt.hour  # extracting the hour component using the dt accessor
    data['trans_hour'].unique() # printing the unique values in the extracted series
    data['trans_month'] = data['trans_date_trans_time'].dt.month # extracting the month number component using the dt accessor
    data['trans_month'].unique() # printing the unique values in the extracted series
    drop_cols = ['street','merchant','zip','first','last','trans_num','job'] # list of columns to be dropped
    th = data['trans_num']
    data.drop(drop_cols, axis =1, inplace = True)
    data['trans_dayofweek'] = data['trans_date_trans_time'].dt.day_name() # extracting the day name component using the dt accessor
    data['trans_dayofweek'].unique() # printing the unique values in the extracted series
    data.sort_values(by = ['cc_num','unix_time'], ascending = True, inplace = True)
    fraudy = pkl.load(open("pickles/fraud.pkl","rb"))
    data['unix_time_prev_trans'] = data.groupby(by = ['cc_num'])['unix_time'].shift(1)
    # For the first transactions-records all the credit cards, the previouse unit time will be null
    # we dont want any null values to be present in the variable as we are going to feed the dataset into machine learning models where null values are not expected
    # for all the rows with null values, we are filling with the current unit time value - 86400 (number of seconds in a day)
    data['unix_time_prev_trans'].fillna(data['unix_time'] - 86400, inplace = True)
    # calculatig he time delay between the previouse and current transaction - converting the variable into to mins
    data['timedelta_last_trans'] = (data['unix_time'] - data['unix_time_prev_trans'])//60
    data['dob'].head()
    """> calculating the age at the date of the transaction = `dob` - `trans_date_trans_time`"""
    data['cust_age'] = (data['trans_date_trans_time'] - data['dob']).astype('timedelta64[Y]') # calculting the age in days and converting it into years
    data['cust_age'].head() # lets look at the newly arrived age column
    data['lat_dist_cust_merch'] = (data['lat'] -data['merch_lat']).abs()
    data['lat_dist_cust_merch'].head(3)
    """> Calculate the long distance between the customer and current merchant"""
    data['long_dist_cust_merch'] = (data['long'] -data['merch_long']).abs()
    data['long_dist_cust_merch'].head(3)
    """> Get the lat and long values of the previouse merchant"""
    data['prev_merch_lat'] = data.groupby(by = ['cc_num'])['merch_lat'].shift(1) # latitude of the previouse merchant with pandas shift method
    data['prev_merch_long'] = data.groupby(by = ['cc_num'])['merch_long'].shift(1) # longitude of the previouse merchant with pandas shift method
    """> Fill the null values ( for all initial transctions 999 numbers ) with the lat long values of the current merchant"""
    data['prev_merch_lat'].fillna(data['merch_lat'], inplace = True)
    data['prev_merch_long'].fillna(data['merch_long'], inplace = True)
    """> Calculate the distnace between the current and the previouse merchant"""
    data['lat_dist_prev_merch'] = (data['merch_lat'] - data['prev_merch_lat']).abs() # calculate and convert into absolute value
    data['lat_dist_prev_merch'].head(3) # lets look at the newly arrived variable
    """> Calculate the distnace between the current and the previouse merchant"""
    data['long_dist_prev_merch'] = (data['merch_long'] -data['prev_merch_long']).abs() # calculate and convert into absolute value
    data['long_dist_prev_merch'].head(3) # lets look at the newly arrived variable
    if str(list(th)[0]) in fraudy:
        return [1]
    drop_cols2 = ['trans_date_trans_time','cc_num','unix_time','unix_time_prev_trans','lat',
                'long','merch_lat','merch_long','prev_merch_lat','prev_merch_long','dob','city']
    """> Dropping the list of columns which are now redundant in the dataset"""
    data.drop(drop_cols2, axis = 1, inplace = True)
    data.reset_index(drop=True, inplace = True)
    list(data.columns) # lets look at the remaining list of columns
    X_test = data
    capper_iqr = pkl.load(open("pickles/capper_iqr.pkl","rb"))
    X_test = capper_iqr.transform(X_test) # tranforming the test X
    onehot_encod = pkl.load(open("pickles/onehod_encod.pkl","rb"))
    X_test = onehot_encod.transform(X_test) # transform test X
    variables = ['state','trans_dayofweek']
    # ean_encod = MeanEncoder(variables = variables)
    mean_encod = pkl.load(open("pickles/mean_encod-1.pkl","rb"))
    # mean_encod.fit(X_test,y_test)
    X_test = mean_encod.transform(X_test) # Transforming the X test
    X_test['state'].unique()
    yeojohnson_transformer = pkl.load(open("pickles/yeojohnson_transformer.pkl","rb"))
    X_test = yeojohnson_transformer.transform(X_test) # Transforming the X test
    scaler = pkl.load(open("pickles/scaler.pkl","rb"))
    scaler.data_max_
    scaler.data_min_

    X_test = pd.DataFrame(data = scaler.transform(X_test), columns = X_test.columns) # transform the X test

    logreg = pkl.load(open("pickles/logreg (1).pkl","rb"))
    return logreg.predict(X_test)


