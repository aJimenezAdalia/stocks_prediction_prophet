'''
1. How to load/save a Prophet model:

# Load model:
with open('path/model_name.json', 'r') as fin:
    model = model_from_json(json.load(fin))

# Save model:
with open('path/model_name.json', 'w') as fout:
    json.dump(model_to_json(model_name), fout) 

-----------------------------------------------------------------------------------------------------------------------

2. Checking training of a model:

Every trained model has a "history" method, that returns the DataFrame that the model trained with.

# Example

a) Loading the model:
with open('models/model_name.json', 'r') as fin:
    model = model_from_json(json.load(fin))

b) Checking the training DataFrame:
train_df = model.history()

'''


# Imports
import pandas as pd
from fbprophet import Prophet
from fbprophet.serialize import model_to_json, model_from_json
import json
import plotly.express as px


# Function to train and return a model
def train(stock_name=None, path_df=None, show_validation_plot=False, future=False, period=False):
    """Train and save a model.

    Args:
        - stock_name: name of the stock to name the model
        - path_df: DataFrame path
        - show_validation_plot: whether the function plot the training results or not
        - future: True means that we'll train the model to make future forecasting
        - period: number of train days (if False, train with all the available data)

    Returns:
        - Save a trained model with the received data

    """

    # Loading the data
    df = pd.read_csv(path_df)
    df.ds = df.ds.astype('datetime64')
    if period:
        df = df[-period:]

    # Training and validation data
    test_size=0.05
    if stock_name == 'bitcoin':
        test_size = 0.025  # crypto market volatility is very high
    test_rows = int(len(df)*test_size)
    train_data = df[:-test_rows].reset_index(drop=True)
    test_data = df[-test_rows:].reset_index(drop=True)

    # Training model in train data
    future_df = test_data[['ds']]
    model = Prophet()
    # Evaluating future
    if future:
        model.fit(df)
    else:
        model.fit(train_data[int(len(train_data)/2):])

    forecast = model.predict(future_df)

    if show_validation_plot:
        # Merging the data to plot predictions and validation data 
        test = forecast.merge(test_data, on='ds', how='right')
        test = test[['ds', 'yhat', 'y']]
        test.columns = ['Date', 'Predictions', 'Actual Price']
        test = test.tail(800) # we don't want to show all the data

        fig = px.scatter(
                    test, x=test.Date, y= test.columns[1:], 
                    color_discrete_sequence=['#0075CF', '#029F65'], 
                    title=f'{stock_name.upper()} - Actual Price vs Predictions')
        fig.update_traces(marker_size=5)
        fig.show()
    
    return model


# Function to save a model
def save_model(save_path, model, stock_name):
    '''Save a provided model in a provided path
    
    Args:
        - save_path: path where the model will be stored
        - model: FB Prophet model

    '''
    
    # Saving the model
    if save_path[-1] not in ['/', '\\']:
        save_path = save_path + '/'
    model_name = stock_name + '.json'
    full_path = save_path + model_name
    with open(full_path, 'w') as fout:
        json.dump(model_to_json(model), fout)
    print(f'Model Succesfully Saved in: \n{full_path}')


# Function lo load a model
def load_model(model_full_path):
    '''Load a pre-trained model from a provided path. 
    The path must contain the model name.
    
    Args:
        - model_full_path: full path where the model exists, model included
        
    Return:
        - model: pre-trained model 
    '''

    # Loading model:
    with open(model_full_path, 'r') as fin:
        model = model_from_json(json.load(fin))

    return model
