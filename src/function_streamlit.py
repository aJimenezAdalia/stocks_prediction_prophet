'''
Streamlit functionality.

'''

# Imports
import streamlit as st
import pandas as pd
from PIL import Image
from utils import make_predictions, predict_with_trained_model



# Main page settings -----------------------------------------------------------------------------------------------------
def config_page():
    st.set_page_config(
        page_title='Stock Predictions', 
        layout='wide'
    )

# Cache
st.cache(suppress_st_warning=True)

# Loading data
def load_data(path):
    df = pd.read_csv(path)
    return df


# FUTURE PREDICTIONS - No data ----------------------------------------------------------------------------------------------
'''
Every stock has their own functionality.

The predictions made within this functions are running on the spot, with models using all the available data.
It shows a Plotly plot to show the results.

Predictions are made from 26/11/2021 to the future.
'''


##########################################################################################################################
# HOME ###################################################################################################################
##########################################################################################################################


'''
Streamlit Main page. Two sidebars and an image.

'''
def home():
    # Title
    st.markdown('# Stock Predictions with Facebook Prophet')
    # Loading the image
    img = Image.open('image_streamlit.jpeg')
    st.image(img, use_column_width='auto')
    st.markdown('#### 6 available stocks:\n- Amazon\n- Apple\n- Bitcoin\n- Ford\n- Microsoft\n- Tesla')



##########################################################################################################################
# VALIDATED PREDICTIONS ##################################################################################################
##########################################################################################################################

'''
One function each stock.

We make predictions with validation data. The predictions are showed with and without validation, with two Plotly plots.

'''

# Available periods to predict
periods_str = (30, 90, 180, 365)
'''
1 month, 3 months, 6 months, 1 year
'''


# 1. AMAZON
def amazon():

    # Variables
    amazon_model_path = 'models/amazon_2021.json'
    amazon_data_path = 'data/amazon.csv'
    stock_name = 'amazon'


    st.markdown('# **Amazon - AMZN**')    
    st.markdown('## CHOOSE A PERIOD IN THE LEFT BAR TO MAKE PREDICTIONS')
    st.markdown('#### - *Predictions are made on the spot with variable data, depending of the selected period*')
    st.markdown('#### - *Models have never seen before the data they are going to predict*')
    st.markdown('#### - *Predictions have upper and lower limits, which indicates that the price will be inside \
    the area in the future.*')

    # Prediction periods
    prediction_period = st.sidebar.radio(
        'Live training', ('Period to Predict','30', '90', '180', '365'))
    predict_trained = st.sidebar.radio(
        'Pre-Trained Model', ('Period to Predict','30', '90', '180', '365'))

    # PERIOD: 1 month
    if prediction_period == '30':
        st.markdown('## AMAZON - 1 Month Predictions with Validation Data')
        st.markdown('### **Live Train and Predictions**')
        # Show predictions - 1 month
        fig1, fig2 = make_predictions(amazon_data_path, 30, False, 'line', True)
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

    if predict_trained == '30':
        st.markdown('### **Pre-Trained Model - Live Predictions**')
        # Show predictions - 1 month
        fig = predict_with_trained_model(amazon_model_path, stock_name, amazon_data_path, 30)
        st.plotly_chart(fig)
    
    # PERIOD: 3 months
    if prediction_period == '90':
        st.markdown('## AMAZON - 3 Months Predictions with Validation Data')
        st.markdown('### **Live Train and Predictions**')
        # Show predictions - 3 months
        fig1, fig2 = make_predictions(amazon_data_path, 90, False, 'line', True)
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

    if predict_trained == '90':
        st.markdown('### **Pre-Trained Model - Live Predictions**')
        # Show predictions - 3 months
        fig = predict_with_trained_model(amazon_model_path, stock_name, amazon_data_path, 90)
        st.plotly_chart(fig)

    # PERIOD: 6 months
    if prediction_period == '180':
        st.markdown('## AMAZON - 6 Months Predictions with Validation Data')
        st.markdown('### **Live Train and Predictions**')
        # Show predictions - 6 months
        fig1, fig2 = make_predictions(amazon_data_path, 180, False, 'line', True)
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

    if predict_trained == '180':
        st.markdown('### **Pre-Trained Model - Live Predictions**')
        # Show predictions - 6 months
        fig = predict_with_trained_model(amazon_model_path, stock_name, amazon_data_path, 180)
        st.plotly_chart(fig)

    # PERIOD: 12 months
    if prediction_period == '365':
        st.markdown('## AMAZON - 1 Year Predictions with Validation Data')
        st.markdown('### **Live Train and Predictions**')
        # Show predictions - 12 months
        fig1, fig2 = make_predictions(amazon_data_path, 365, False, 'line', True)
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

    if predict_trained == '365':
        st.markdown('### **Pre-Trained Model - Live Predictions**')
        # Show predictions - 12 months
        fig = predict_with_trained_model(amazon_model_path, stock_name, amazon_data_path, 365)
        st.plotly_chart(fig)

# 2. APPLE
def apple():
    # Variables
    apple_model_path = 'models/apple_2021.json'
    apple_data_path = 'data/apple.csv'
    stock_name = 'apple'

    st.markdown('# **Apple - AAPL**')    
    st.markdown('## CHOOSE A PERIOD IN THE LEFT BAR TO MAKE PREDICTIONS')
    st.markdown('#### - *Predictions are made on the spot with variable data, depending of the selected period*')
    st.markdown('#### - *Models have never seen before the data they are going to predict*')
    st.markdown('#### - *Predictions have upper and lower limits, which indicates that the price will be inside \
    the area in the future.*')

    # Prediction periods
    prediction_period = st.sidebar.radio(
        'Live training', ('Period to Predict','30', '90', '180', '365'))
    predict_trained = st.sidebar.radio(
        'Pre-Trained Model', ('Period to Predict','30', '90', '180', '365'))


    # PERIOD: 1 month
    if prediction_period == '30':
        st.markdown('## APPLE - 1 Month Predictions with Validation Data')
        st.markdown('### **Live Train and Predictions**')
        # Show predictions - 1 month
        fig1, fig2 = make_predictions(apple_data_path, 30, False, 'line', True)
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

    if predict_trained == '30':
        st.markdown('### **Pre-Trained Model - Live Predictions**')
        # Show predictions - 1 month
        fig = predict_with_trained_model(apple_model_path, stock_name, apple_data_path, 30)
        st.plotly_chart(fig)
    
    # PERIOD: 3 months
    if prediction_period == '90':
        st.markdown('## APPLE - 3 Months Predictions with Validation Data')
        st.markdown('### **Live Train and Predictions**')
        # Show predictions - 3 months
        fig1, fig2 = make_predictions(apple_data_path, 90, False, 'line', True)
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

    if predict_trained == '90':
        st.markdown('### **Pre-Trained Model - Live Predictions**')
        # Show predictions - 3 months
        fig = predict_with_trained_model(apple_model_path, stock_name, apple_data_path, 90)
        st.plotly_chart(fig)

    # PERIOD: 6 months
    if prediction_period == '180':
        st.markdown('## APPLE - 6 Months Predictions with Validation Data')
        st.markdown('### **Live Train and Predictions**')
        # Show predictions - 6 months
        fig1, fig2 = make_predictions(apple_data_path, 180, False, 'line', True)
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

    if predict_trained == '180':
        st.markdown('### **Pre-Trained Model - Live Predictions**')
        # Show predictions - 6 months
        fig = predict_with_trained_model(apple_model_path, stock_name, apple_data_path, 180)
        st.plotly_chart(fig)

    # PERIOD: 12 months
    if prediction_period == '365':
        st.markdown('## APPLE - 1 Year Predictions with Validation Data')
        st.markdown('### **Live Train and Predictions**')
        # Show predictions - 12 months
        fig1, fig2 = make_predictions(apple_data_path, 365, False, 'line', True)
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

    if predict_trained == '365':
        st.markdown('### **Pre-Trained Model - Live Predictions**')
        # Show predictions - 12 months
        fig = predict_with_trained_model(apple_model_path, stock_name, apple_data_path, 365)
        st.plotly_chart(fig)


# 3. BTC-USD
def bitcoin():
    # Variables
    btc_model_path = 'models/bitcoin_2021.json'
    btc_data_path = 'data/bitcoin.csv'
    stock_name = 'BTC-USD'

    st.markdown('# **Bitcoin  BTC-USD**')    
    st.markdown('## CHOOSE A PERIOD IN THE LEFT BAR TO MAKE PREDICTIONS')
    st.markdown('#### - *Predictions are made on the spot with variable data, depending of the selected period*')
    st.markdown('#### - *Models have never seen before the data they are going to predict*')
    st.markdown('#### - *Predictions have upper and lower limits, which indicates that the price will be inside \
    the area in the future.*')

    # Prediction periods
    prediction_period = st.sidebar.radio(
        'Live training', ('Period to Predict','30', '90', '180', '365'))
    predict_trained = st.sidebar.radio(
        'Pre-Trained Model', ('Period to Predict','30', '90', '180', '365'))
   

    # PERIOD: 1 month
    if prediction_period == '30':
        st.markdown('## BTC-USD - 1 Month Predictions with Validation Data')
        st.markdown('### **Live Train and Predictions**')
        # Show predictions - 1 month
        fig1, fig2 = make_predictions(btc_data_path, 30, False, 'line', True)
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

    if predict_trained == '30':
        st.markdown('### **Pre-Trained Model - Live Predictions**')
        # Show predictions - 1 month
        fig = predict_with_trained_model(btc_model_path, stock_name, btc_data_path, 30)
        st.plotly_chart(fig)
    
    # PERIOD: 3 months
    if prediction_period == '90':
        st.markdown('## BTC-USD - 3 Months Predictions with Validation Data')
        st.markdown('### **Live Train and Predictions**')
        # Show predictions - 3 months
        fig1, fig2 = make_predictions(btc_data_path, 90, False, 'line', True)
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

    if predict_trained == '90':
        st.markdown('### **Pre-Trained Model - Live Predictions**')
        # Show predictions - 3 months
        fig = predict_with_trained_model(btc_model_path, stock_name, btc_data_path, 90)
        st.plotly_chart(fig)

    # PERIOD: 6 months
    if prediction_period == '180':
        st.markdown('## BTC-USD - 6 Months Predictions with Validation Data')
        st.markdown('### **Live Train and Predictions**')
        # Show predictions - 6 months
        fig1, fig2 = make_predictions(btc_data_path, 180, False, 'line', True)
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

    if predict_trained == '180':
        st.markdown('### **Pre-Trained Model - Live Predictions**')
        # Show predictions - 6 months
        fig = predict_with_trained_model(btc_model_path, stock_name, btc_data_path, 180)
        st.plotly_chart(fig)

    # PERIOD: 12 months
    if prediction_period == '365':
        st.markdown('## BTC-USD - 1 Year Predictions with Validation Data')
        st.markdown('### **Live Train and Predictions**')
        # Show predictions - 12 months
        fig1, fig2 = make_predictions(btc_data_path, 365, False, 'line', True)
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

    if predict_trained == '365':
        st.markdown('### **Pre-Trained Model - Live Predictions**')
        # Show predictions - 12 months
        fig = predict_with_trained_model(btc_model_path, stock_name, btc_data_path, 365)
        st.plotly_chart(fig)



# 4. FORD
def ford():
    # Variables
    ford_model_path = 'models/ford_2021.json'
    ford_data_path = 'data/ford.csv'
    stock_name = 'ford'

    st.markdown('# **FORD - NYSE: F**')    
    st.markdown('## CHOOSE A PERIOD IN THE LEFT BAR TO MAKE PREDICTIONS')
    st.markdown('#### - *Predictions are made on the spot with variable data, depending of the selected period*')
    st.markdown('#### - *Models have never seen before the data they are going to predict*')
    st.markdown('#### - *Predictions have upper and lower limits, which indicates that the price will be inside \
    the area in the future.*')

    # Prediction periods
    prediction_period = st.sidebar.radio(
        'Live training', ('Period to Predict','30', '90', '180', '365'))
    predict_trained = st.sidebar.radio(
        'Pre-Trained Model', ('Period to Predict','30', '90', '180', '365'))
   

    # PERIOD: 1 month
    if prediction_period == '30':
        st.markdown('## FORD - 1 Month Predictions with Validation Data')
        st.markdown('### **Live Train and Predictions**')
        # Show predictions - 1 month
        fig1, fig2 = make_predictions(ford_data_path, 30, False, 'line', True)
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

    if predict_trained == '30':
        st.markdown('### **Pre-Trained Model - Live Predictions**')
        # Show predictions - 1 month
        fig = predict_with_trained_model(ford_model_path, stock_name, ford_data_path, 30)
        st.plotly_chart(fig)
    
    # PERIOD: 3 months
    if prediction_period == '90':
        st.markdown('## FORD - 3 Months Predictions with Validation Data')
        st.markdown('### **Live Train and Predictions**')
        # Show predictions - 3 months
        fig1, fig2 = make_predictions(ford_data_path, 90, False, 'line', True)
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

    if predict_trained == '90':
        st.markdown('### **Pre-Trained Model - Live Predictions**')
        # Show predictions - 3 months
        fig = predict_with_trained_model(ford_model_path, stock_name, ford_data_path, 90)
        st.plotly_chart(fig)

    # PERIOD: 6 months
    if prediction_period == '180':
        st.markdown('## FORD - 6 Months Predictions with Validation Data')
        st.markdown('### **Live Train and Predictions**')
        # Show predictions - 6 months
        fig1, fig2 = make_predictions(ford_data_path, 180, False, 'line', True)
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

    if predict_trained == '180':
        st.markdown('### **Pre-Trained Model - Live Predictions**')
        # Show predictions - 6 months
        fig = predict_with_trained_model(ford_model_path, stock_name, ford_data_path, 180)
        st.plotly_chart(fig)

    # PERIOD: 12 months
    if prediction_period == '365':
        st.markdown('## FORD - 1 Year Predictions with Validation Data')
        st.markdown('### **Live Train and Predictions**')
        # Show predictions - 12 months
        fig1, fig2 = make_predictions(ford_data_path, 365, False, 'line', True)
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

    if predict_trained == '365':
        st.markdown('### **Pre-Trained Model - Live Predictions**')
        # Show predictions - 12 months
        fig = predict_with_trained_model(ford_model_path, stock_name, ford_data_path, 365)
        st.plotly_chart(fig)


# 5. MICROSOFT
def microsoft():
    # Variables
    ms_model_path = 'models/microsoft_2021.json'
    ms_data_path = 'data/microsoft.csv'
    stock_name = 'microsoft'

    st.markdown('# **MICROSOFT - MSFT**')    
    st.markdown('## CHOOSE A PERIOD IN THE LEFT BAR TO MAKE PREDICTIONS')
    st.markdown('#### - *Predictions are made on the spot with variable data, depending of the selected period*')
    st.markdown('#### - *Models have never seen before the data they are going to predict*')
    st.markdown('#### - *Predictions have upper and lower limits, which indicates that the price will be inside \
    the area in the future.*')

    # Prediction periods
    prediction_period = st.sidebar.radio(
        'Live training', ('Period to Predict','30', '90', '180', '365'))
    predict_trained = st.sidebar.radio(
        'Pre-Trained Model', ('Period to Predict','30', '90', '180', '365'))
   

    # PERIOD: 1 month
    if prediction_period == '30':
        st.markdown('## MICROSOFT - 1 Month Predictions with Validation Data')
        st.markdown('### **Live Train and Predictions**')
        # Show predictions - 1 month
        fig1, fig2 = make_predictions(ms_data_path, 30, False, 'line', True)
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

    if predict_trained == '30':
        st.markdown('### **Pre-Trained Model - Live Predictions**')
        # Show predictions - 1 month
        fig = predict_with_trained_model(ms_model_path, stock_name, ms_data_path, 30)
        st.plotly_chart(fig)
    
    # PERIOD: 3 months
    if prediction_period == '90':
        st.markdown('## MICROSOFT - 3 Months Predictions with Validation Data')
        st.markdown('### **Live Train and Predictions**')
        # Show predictions - 3 months
        fig1, fig2 = make_predictions(ms_data_path, 90, False, 'line', True)
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

    if predict_trained == '90':
        st.markdown('### **Pre-Trained Model - Live Predictions**')
        # Show predictions - 3 months
        fig = predict_with_trained_model(ms_model_path, stock_name, ms_data_path, 90)
        st.plotly_chart(fig)

    # PERIOD: 6 months
    if prediction_period == '180':
        st.markdown('## MICROSOFT - 6 Months Predictions with Validation Data')
        st.markdown('### **Live Train and Predictions**')
        # Show predictions - 6 months
        fig1, fig2 = make_predictions(ms_data_path, 180, False, 'line', True)
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

    if predict_trained == '180':
        st.markdown('### **Pre-Trained Model - Live Predictions**')
        # Show predictions - 6 months
        fig = predict_with_trained_model(ms_model_path, stock_name, ms_data_path, 180)
        st.plotly_chart(fig)

    # PERIOD: 12 months
    if prediction_period == '365':
        st.markdown('## MICROSOFT - 1 Year Predictions with Validation Data')
        st.markdown('### **Live Train and Predictions**')
        # Show predictions - 12 months
        fig1, fig2 = make_predictions(ms_data_path, 365, False, 'line', True)
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

    if predict_trained == '365':
        st.markdown('### **Pre-Trained Model - Live Predictions**')
        # Show predictions - 12 months
        fig = predict_with_trained_model(ms_model_path, stock_name, ms_data_path, 365)
        st.plotly_chart(fig)

# 6. TESLA
def tesla():
    # Variables
    tesla_model_path = 'models/tesla_2021.json'
    tesla_data_path = 'data/tesla.csv'
    stock_name = 'tesla'

    st.markdown('# **TESLA - TSLA**')    
    st.markdown('## CHOOSE A PERIOD IN THE LEFT BAR TO MAKE PREDICTIONS')
    st.markdown('#### - *Predictions are made on the spot with variable data, depending of the selected period*')
    st.markdown('#### - *Models have never seen before the data they are going to predict*')
    st.markdown('#### - *Predictions have upper and lower limits, which indicates that the price will be inside \
    the area in the future.*')

    # Prediction periods
    prediction_period = st.sidebar.radio(
        'Live training', ('Period to Predict','30', '90', '180', '365'))
    predict_trained = st.sidebar.radio(
        'Pre-Trained Model', ('Period to Predict','30', '90', '180', '365'))
   

    # PERIOD: 1 month
    if prediction_period == '30':
        st.markdown('## TESLA - 1 Month Predictions with Validation Data')
        st.markdown('### **Live Train and Predictions**')
        # Show predictions - 1 month
        fig1, fig2 = make_predictions(tesla_data_path, 30, False, 'line', True)
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

    if predict_trained == '30':
        st.markdown('### **Pre-Trained Model - Live Predictions**')
        # Show predictions - 1 month
        fig = predict_with_trained_model(tesla_model_path, stock_name, tesla_data_path, 30)
        st.plotly_chart(fig)
    
    # PERIOD: 3 months
    if prediction_period == '90':
        st.markdown('## TESLA - 3 Months Predictions with Validation Data')
        st.markdown('### **Live Train and Predictions**')
        # Show predictions - 3 months
        fig1, fig2 = make_predictions(tesla_data_path, 90, False, 'line', True)
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

    if predict_trained == '90':
        st.markdown('### **Pre-Trained Model - Live Predictions**')
        # Show predictions - 3 months
        fig = predict_with_trained_model(tesla_model_path, stock_name, tesla_data_path, 90)
        st.plotly_chart(fig)

    # PERIOD: 6 months
    if prediction_period == '180':
        st.markdown('## TESLA - 6 Months Predictions with Validation Data')
        st.markdown('### **Live Train and Predictions**')
        # Show predictions - 6 months
        fig1, fig2 = make_predictions(tesla_data_path, 180, False, 'line', True)
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

    if predict_trained == '180':
        st.markdown('### **Pre-Trained Model - Live Predictions**')
        # Show predictions - 6 months
        fig = predict_with_trained_model(tesla_model_path, stock_name, tesla_data_path, 180)
        st.plotly_chart(fig)

    # PERIOD: 12 months
    if prediction_period == '365':
        st.markdown('## TESLA - 1 Year Predictions with Validation Data')
        st.markdown('### **Live Train and Predictions**')
        # Show predictions - 12 months
        fig1, fig2 = make_predictions(tesla_data_path, 365, False, 'line', True)
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

    if predict_trained == '365':
        st.markdown('### **Pre-Trained Model - Live Predictions**')
        # Show predictions - 12 months
        fig = predict_with_trained_model(tesla_model_path, stock_name, tesla_data_path, 365)
        st.plotly_chart(fig)


def val_predictions():

    menu3 = st.sidebar.selectbox('Validated Predictions', 
    ['Amazon', 'Apple', 'BTC-USD', 'Ford', 'Microsoft', 'Tesla']) 
    
    if menu3 == 'Amazon':
        amazon()
    elif menu3 == 'Apple':
        apple()
    elif menu3 == 'BTC-USD':
        bitcoin()
    elif menu3 == 'Ford':
        ford()   
    elif menu3 == 'Microsoft':
        microsoft() 
    elif menu3 == 'Tesla':
        tesla()



##########################################################################################################################
# FUTURE FORECASTING #####################################################################################################
##########################################################################################################################

# Available periods to predict
future_periods = ['Choose a period', '1 month', '3 months', '6 months', '1 year']

# 1. AMAZON - Future
def amazon_future():
    st.markdown('## CHOOSE A PERIOD IN THE LEFT BAR TO MAKE PREDICTIONS')
    st.markdown('#### - *Predictions are made on the spot with pre-trained models with all available data*')
    st.markdown('#### - *Models have never seen before the data they are going to predict*')
    st.markdown('#### - *Predictions have upper and lower limits, which indicates that the price will be inside \
    the area in the future.*')

    # Prediction periods
    future_period = st.sidebar.selectbox(
        'Choose a period to predict', future_periods)

    # Amazon data path
    amzn_path = 'data/amazon.csv'

    # PERIOD: 1 month
    if future_period == '1 month':
        fig = make_predictions(amzn_path, 30, True, 'scatter', True)
        st.plotly_chart(fig)
    # PERIOD: 3 months
    elif future_period == '3 months':
        fig = make_predictions(amzn_path, 90, True, 'scatter', True)
        st.plotly_chart(fig)
    # PERIOD: 6 months
    elif future_period == '6 months':
        fig = make_predictions(amzn_path, 180, True, 'scatter', True)
        st.plotly_chart(fig)
    # PERIOD: 1 year
    elif future_period == '1 year':
        fig = make_predictions(amzn_path, 365, True, 'scatter', True)
        st.plotly_chart(fig)


# 2. APPLE - Future
def apple_future():
    st.markdown('## CHOOSE A PERIOD IN THE LEFT BAR TO MAKE PREDICTIONS')
    st.markdown('#### - *Predictions are made on the spot with pre-trained models with all available data*')
    st.markdown('#### - *Models have never seen before the data they are going to predict*')
    st.markdown('#### - *Predictions have upper and lower limits, which indicates that the price will be inside \
    the area in the future.*')

    # Prediction periods
    future_period = st.sidebar.selectbox(
        'Choose a period to predict', future_periods)

    # Apple data path
    apple_path = 'data/apple.csv'

    # PERIOD: 1 month
    if future_period == '1 month':
        fig = make_predictions(apple_path, 30, True, 'scatter', True)
        st.plotly_chart(fig)
    # PERIOD: 3 months
    elif future_period == '3 months':
        fig = make_predictions(apple_path, 90, True, 'scatter', True)
        st.plotly_chart(fig)
    # PERIOD: 6 months
    elif future_period == '6 months':
        fig = make_predictions(apple_path, 180, True, 'scatter', True)
        st.plotly_chart(fig)
    # PERIOD: 1 year
    elif future_period == '1 year':
        fig = make_predictions(apple_path, 365, True, 'scatter', True)
        st.plotly_chart(fig)

# 3. BITCOIN - Future
def btc_future():
    st.markdown('## CHOOSE A PERIOD IN THE LEFT BAR TO MAKE PREDICTIONS')
    st.markdown('#### - *Predictions are made on the spot with pre-trained models with all available data*')
    st.markdown('#### - *Models have never seen before the data they are going to predict*')
    st.markdown('#### - *Predictions have upper and lower limits, which indicates that the price will be inside \
    the area in the future.*')

    # Prediction periods
    future_period = st.sidebar.selectbox(
        'Choose a period to predict', future_periods)

    # BTC data path
    btc_path = 'data/bitcoin.csv'

    # PERIOD: 1 month
    if future_period == '1 month':
        fig = make_predictions(btc_path, 30, True, 'scatter', True)
        st.plotly_chart(fig)
    # PERIOD: 3 months
    elif future_period == '3 months':
        fig = make_predictions(btc_path, 90, True, 'scatter', True)
        st.plotly_chart(fig)
    # PERIOD: 6 months
    elif future_period == '6 months':
        fig = make_predictions(btc_path, 185, True, 'scatter', True)
        st.plotly_chart(fig)
    # PERIOD: 1 year
    elif future_period == '1 year':
        fig = make_predictions(btc_path, 365, True, 'scatter', True)
        st.plotly_chart(fig)


# 4. FORD - Future
def ford_future():
    st.markdown('## CHOOSE A PERIOD IN THE LEFT BAR TO MAKE PREDICTIONS')
    st.markdown('#### - *Predictions are made on the spot with pre-trained models with all available data*')
    st.markdown('#### - *Models have never seen before the data they are going to predict*')
    st.markdown('#### - *Predictions have upper and lower limits, which indicates that the price will be inside \
    the area in the future.*')

    # Prediction periods
    future_period = st.sidebar.selectbox(
        'Choose a period to predict', future_periods)

    # Ford data path
    ford_path = 'data/ford.csv'

    # PERIOD: 1 month
    if future_period == '1 month':
        fig = make_predictions(ford_path, 30, True, 'scatter', True)
        st.plotly_chart(fig)
    # PERIOD: 3 months
    elif future_period == '3 months':
        fig = make_predictions(ford_path, 90, True, 'scatter', True)
        st.plotly_chart(fig)
    # PERIOD: 6 months
    elif future_period == '6 months':
        fig = make_predictions(ford_path, 185, True, 'scatter', True)
        st.plotly_chart(fig)
    # PERIOD: 1 year
    elif future_period == '1 year':
        fig = make_predictions(ford_path, 365, True, 'scatter', True)
        st.plotly_chart(fig)


# 5. MICROSOFT - Future
def microsoft_future():
    st.markdown('## CHOOSE A PERIOD IN THE LEFT BAR TO MAKE PREDICTIONS')
    st.markdown('#### - *Predictions are made on the spot with pre-trained models with all available data*')
    st.markdown('#### - *Models have never seen before the data they are going to predict*')
    st.markdown('#### - *Predictions have upper and lower limits, which indicates that the price will be inside \
    the area in the future.*')

    # Prediction periods
    future_period = st.sidebar.selectbox(
        'Choose a period to predict', future_periods)

    # Ford data path
    microsoft_path = 'data/microsoft.csv'

    # PERIOD: 1 month
    if future_period == '1 month':
        fig = make_predictions(microsoft_path, 30, True, 'scatter', True)
        st.plotly_chart(fig)
    # PERIOD: 3 months
    elif future_period == '3 months':
        fig = make_predictions(microsoft_path, 90, True, 'scatter', True)
        st.plotly_chart(fig)
    # PERIOD: 6 months
    elif future_period == '6 months':
        fig = make_predictions(microsoft_path, 180, True, 'scatter', True)
        st.plotly_chart(fig)
    # PERIOD: 1 year
    elif future_period == '1 year':
        fig = make_predictions(microsoft_path, 365, True, 'scatter', True)
        st.plotly_chart(fig)


# 6. TESLA - Future
def tesla_future():
    st.markdown('## CHOOSE A PERIOD IN THE LEFT BAR TO MAKE PREDICTIONS')
    st.markdown('#### - *Predictions are made on the spot with pre-trained models with all available data*')
    st.markdown('#### - *Models have never seen before the data they are going to predict*')
    st.markdown('#### - *Predictions have upper and lower limits, which indicates that the price will be inside \
    the area in the future.*')

    # Prediction periods
    future_period = st.sidebar.selectbox(
        'Choose a period to predict', future_periods)

    # Ford data path
    tesla_path = 'data/tesla.csv'

    # PERIOD: 1 month
    if future_period == '1 month':
        fig = make_predictions(tesla_path, 30, True, 'scatter', True)
        st.plotly_chart(fig)
    # PERIOD: 3 months
    elif future_period == '3 months':
        fig = make_predictions(tesla_path, 90, True, 'scatter', True)
        st.plotly_chart(fig)
    # PERIOD: 6 months
    elif future_period == '6 months':
        fig = make_predictions(tesla_path, 180, True, 'scatter', True)
        st.plotly_chart(fig)
    # PERIOD: 1 year
    elif future_period == '1 year':
        fig = make_predictions(tesla_path, 365, True, 'scatter', True)
        st.plotly_chart(fig)



 # Sidebar 2 - Future

def future():
    menu2 = st.sidebar.selectbox('Future Predictions', 
    ['Choose a stock', 'Amazon', 'Apple', 'BTC-USD', 'Ford', 'Microsoft', 'Tesla'])   

    if menu2 == 'Amazon':
        st.legacy_caching.clear_cache()
        amazon_future()
    elif menu2 == 'Apple':
        apple_future()
    elif menu2 == 'BTC-USD':
        btc_future()
    elif menu2 == 'Ford':
        ford_future()
    elif menu2 == 'Microsoft':
        microsoft_future()
    elif menu2 == 'Tesla':
        tesla_future()


###########################################################################################################################    
# CONCLUSSIONS ############################################################################################################
###########################################################################################################################  


def conclussions():
    st.markdown('# Conclussions')
    st.markdown('')
    st.markdown('### About the models')
    st.markdown('- FB Prophet is a great model to detect trends and seasonality, and works better in long term.')
    st.markdown('- Easy to use, Plotly and Pandas integred.')
    st.markdown('- Not too much documentation out there, the official docs are not the best.')
    st.markdown('')
    st.markdown('### About the project')
    st.markdown('- Is not possible to predict the stock prices, since there exists too much variables that can impact the prices.')
    st.markdown('- Some stocks tends to follow trends, if you can identify them it is possible to win.')
    for i in range(40):
        st.markdown(' ')
    st.markdown('# Do you want to get RICH in five years? Buy Bitcoin!')
    fig = make_predictions('data/bitcoin.csv', 1800, True, 'scatter', True)
    st.plotly_chart(fig)
