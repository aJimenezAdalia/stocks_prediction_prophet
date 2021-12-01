'''
Utility functions

'''

# Imports
import pandas as pd
import datetime as dt
from datetime import timedelta
import plotly.express as px 
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import seaborn as sns
sns.set(rc={'figure.figsize':(14,10)})
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from train import load_model, save_model


# Data transformation, needed to fit Prophet models
def condition_data(df):
    """Transform a DataFrame to make predictions 
    with Facebook Prophet model.
    
    Args:
        - df: raw DataFrame from Yahoo, via datareader
        
    Return:
        - Properly builded DataFrame to fit FB Prophet models
    """

    # Modifying the data
    df['Date'] = df.index
    df = df[['Date', 'Close']]
    df.columns = ['ds', 'y']
    df.y = df.y.astype('float')
    df.ds = df.ds.astype('datetime64')
    df = df.reset_index(drop=True)

    return df


# Make predictions over a selected period
def make_predictions(df_path, period, future=False, plot_kind='line', return_plots=False):
    """Predict a custom selected period by the user. This function can either 
    predict with or without validation data, using the 'future' parameter.
    
    Args:
        - df_path: DataFrame path
        - period: number of days to predict
        - future: whether the user want to predict the future or with validation data
        - plot_kind: two choices here: 'line' (default) or 'scatter'
        - return_plots: whether the user want the function returns the generated plots

    Return:
        - Predictions Plotly plots

    Raises:
        - ValueError: if plot_kind != 'line' or 'scatter'
    """

    if plot_kind not in ['line', 'scatter']:
        raise ValueError ('plot_kind argument must be "line" or "scatter"')

    # Capturing the stock name to include it into the plot titles
    stock_name = df_path[5:-4].upper()

    # Transforming the period to real days
    actual_period = int(period/1.42) # 1.42 is the average ratio business-days/month
    if stock_name == 'BITCOIN':  # crypto market is available 24/7
        actual_period = period
    # Loading the data
    df = pd.read_csv(df_path)
    df.ds = df.ds.astype('datetime64')

    # Building time periods
    if future:
        end_date = dt.datetime.strptime('2021-11-26', '%Y-%m-%d').date() + timedelta(days=period)
    else:
        end_date = dt.datetime.strptime('2021-11-26', '%Y-%m-%d').date() # last day of available data
    start_date = end_date - timedelta(days=period)

    # Creating future DataFrame - Dates only
    dates = []
    for i in range((end_date - start_date).days):
        dates +=  [(start_date+timedelta(i)).strftime('%Y-%m-%d')]

    dates_df = pd.DataFrame(dates)
    dates_df.columns = ['ds']

    # Train and validation data
    if future:
        X_train = df[-800-actual_period:]
    else:
        X_train = df[-800-actual_period: -actual_period]
    X_test = dates_df

    # Making predictions - training with the last two years of data
    model = Prophet()
    model.fit(X_train)
    forecast = model.predict(X_test)

    # Showing the isolated predictions
    fig1 = plot_plotly(model, forecast, xlabel='Date', ylabel='Price')
    fig1.update_layout(title=f'{stock_name} - Price Predictions')
    if return_plots and future:
        return fig1
    if not return_plots:
        fig1.show()

    # Merging the data to plot predictions and validation data 
    test = forecast.merge(df, on='ds', how='right')
    test = test[['ds', 'yhat', 'y']]
    test.columns = ['Date', 'Predictions', 'Actual Price']
    test = test.tail(800) # we don't want to show all the data

    # Visual comparison between real and predicted values
    if not future:
        if plot_kind == 'line':
            fig2 = px.line(
                test, x=test.Date, y= test.columns[1:], 
                color_discrete_sequence=['#0075CF', '#029F65'], 
                title=f'{stock_name} - Actual Price vs Predictions',
                width=950, height=500)
            fig2.update_traces(marker_size=5)
            if not return_plots:
                fig2.show()
        elif plot_kind == 'scatter':
            fig2 = px.scatter(
                test, x=test.Date, y= test.columns[1:], 
                color_discrete_sequence=['#0075CF', '#029F65'], 
                title=f'{stock_name} - Actual Price vs Predictions',
                width=950, height=500)
            fig2.update_traces(marker_size=5)
            if not return_plots:
                fig2.show()
    
    # Returns —optional
    if return_plots:
        return fig1, fig2


def predict_with_trained_model(model_path, stock_name, stock_data_path, period):
    '''Make predictions with pre-trained model, and 
    return a plot with the validation data.
    
    Args:
        - model_path: full path of the trained model
        - stock_name: name of the stock
        - stock_data_path:  full path of the data
        - period: days to forecast

    Return:
        - Plotly figure with predictions and validation data        
        '''
    # Get Model and Data
    model = load_model(model_path)
    data = pd.read_csv(stock_data_path)
    data['ds'] = data['ds'].astype('datetime64')

    # Data to predict
    X_test = data[-period:][['ds']]

    # Predictions
    forecast = model.predict(X_test)

    # Merging the data to plot results
    val = forecast.merge(data, on='ds', how='right')
    val = val[['ds', 'yhat', 'y']]
    val.columns = ['Date', 'Predicted Price', 'True Price']
    val = val.tail(800)

    # Plot
    fig = px.line(val, x=val.Date, y=val.columns[1:], width=950, height=500, 
                    title=f'{stock_name.upper()} stock Predictions - Pre-Trained Model')
    
    return fig


def train_model_to_streamlit(data_path, stock_name, start_end_train):
    '''Train and save a model specifically to make and show 
    predictions on Streamlit. This models are trained to predict 
    2021 with solid MAE.
    
    Args:
        - data_path: path of the data
        - stock_name: name of the stock 
        - start_end_train: start and end year to train the model (sequence)
    '''

    # Load Data
    data = pd.read_csv(data_path)
    data.ds = data.ds.astype('datetime64')

    # Get training period
    start = start_end_train[0]
    end = start_end_train[1]
    X_train = data[(data.ds.dt.year >= start) & (data.ds.dt.year <= end)]

    # Training model
    model = Prophet()
    model.fit(X_train)

    # Save model
    model_name = stock_name.lower() + '_2021'
    save_model('../models/', model, model_name)