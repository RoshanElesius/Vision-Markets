import pandas as pd
import numpy as np
import math
import datetime as dt
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import LSTM, GRU
from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression 
from itertools import cycle
from statsmodels.tsa.arima.model import ARIMA
from flask import Flask, request, render_template, redirect, url_for, session, jsonify
import sqlite3
import requests


def lstm(bist100):
    #bist100 = pd.read_csv(data)
    # Clean column names (strip any leading/trailing spaces)
    bist100.columns = bist100.columns.str.strip()
    if bist100['Date'].duplicated().sum() > 0:
        print("Found duplicates in the 'Date' column!")
        bist100 = bist100.drop_duplicates(subset=['Date'])
        try:
            bist100['date'] = pd.to_datetime(bist100['Date'], errors='coerce')
            print("Date column successfully converted to datetime!")
        except Exception as e:
            print(f"Error converting date column: {e}")
            print(bist100['date'].head()) 

    bist100.rename(columns={"Date":"date","Open":"open","High":"high","Low":"low","Close":"close"}, inplace= True)
    bist100['date'] = pd.to_datetime(bist100.date)
    bist100.sort_values(by='date', inplace=True)
    bist100["close"] = bist100["close"].replace({'\$': ''}, regex=True)
    bist100["close"] = bist100["close"].apply(pd.to_numeric)
    closedf = bist100[['date','close']]

    close_stock = closedf.copy()
    del closedf['date']
    scaler=MinMaxScaler(feature_range=(0,1))
    closedf=scaler.fit_transform(np.array(closedf).reshape(-1,1))

    training_size=int(len(closedf)*0.65)
    test_size=len(closedf)-training_size
    train_data,test_data=closedf[0:training_size,:],closedf[training_size:len(closedf),:1]
#https://mixkit.co/free-stock-video/stock-market/
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    time_step = 15
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


    tf.keras.backend.clear_session()
    model=Sequential()
    model.add(LSTM(32,return_sequences=True,input_shape=(time_step,1)))
    model.add(LSTM(32,return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])

    model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=1,batch_size=5,verbose=0)

    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
    original_ytest = scaler.inverse_transform(y_test.reshape(-1,1))


    accuracy= r2_score(original_ytest, test_predict)



    # shift train predictions for plotting

    look_back=time_step
    trainPredictPlot = np.empty_like(closedf)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(closedf)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(closedf)-1, :] = test_predict

    names = cycle(['Original close price','Train predicted close price','Test predicted close price'])


    plotdf = pd.DataFrame({'date': close_stock['date'],
                           'original_close': close_stock['close'],
                          'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                          'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})

    fig = px.line(plotdf,x=plotdf['date'], y=[plotdf['original_close'],plotdf['train_predicted_close'],
                                              plotdf['test_predicted_close']],
                  labels={'value':'Stock price','date': 'Date'})
    fig.update_layout(title_text='Comparision between original close price vs predicted close price',
                      plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
    fig.for_each_trace(lambda t:  t.update(name = next(names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.write_image("static/lstm/lstm_train.jpg")



    x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

    from numpy import array

    lst_output=[]
    n_steps=time_step
    i=0
    pred_days = 7
    
    while(i<pred_days):

        if(len(temp_input)>time_step):

            x_input=np.array(temp_input[1:])
            #print("{} day input {}".format(i,x_input))
            x_input = x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))

            yhat = model.predict(x_input, verbose=0)
            #print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)

            lst_output.extend(yhat.tolist())
            i=i+1

        else:

            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())

            lst_output.extend(yhat.tolist())
            i=i+1

    #print("Output of predicted next days: ", len(lst_output))



    last_days=np.arange(1,time_step+1)
    day_pred=np.arange(time_step+1,time_step+pred_days+1)



    temp_mat = np.empty((len(last_days)+pred_days+1,1))
    temp_mat[:] = np.nan
    temp_mat = temp_mat.reshape(1,-1).tolist()[0]

    last_original_days_value = temp_mat
    next_predicted_days_value = temp_mat

    last_original_days_value[0:time_step+1] = scaler.inverse_transform(closedf[len(closedf)-time_step:]).reshape(1,-1).tolist()[0]
    next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

    new_pred_plot = pd.DataFrame({
        'last_original_days_value':last_original_days_value,
        'next_predicted_days_value':next_predicted_days_value
    })

    names = cycle(['Last 15 days close price','Predicted next 7 days close price'])

    fig = px.line(new_pred_plot,x=new_pred_plot.index, y=[new_pred_plot['last_original_days_value'],
                                                          new_pred_plot['next_predicted_days_value']],
                  labels={'value': 'Stock price','index': 'Timestamp'})
    fig.update_layout(title_text='next 7 days',
                      plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')

    fig.for_each_trace(lambda t:  t.update(name = next(names)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.write_image("static/lstm/future_lstm.jpg")


    lstmdf=closedf.tolist()
    lstmdf.extend((np.array(lst_output).reshape(-1,1)).tolist())
    lstmdf=scaler.inverse_transform(lstmdf).reshape(1,-1).tolist()[0]

    names = cycle(['Close price'])

    fig = px.line(lstmdf,labels={'value': 'Stock price','index': 'Timestamp'})
    fig.update_layout(title_text='Plotting whole closing stock price with prediction',
                      plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Stock')

    fig.for_each_trace(lambda t:  t.update(name = next(names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.write_image("static/lstm/final_lstm.jpg")


    predicted_values = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1)).reshape(1, -1).tolist()[0]
    combined_values = list(zip( predicted_values))
    prediction_df = pd.DataFrame(combined_values, columns=[ 'Predicted Close Price'])
    prediction_df = pd.DataFrame(combined_values, columns=[ 'Predicted Close Price'])
    print("lstm code completed")
    return prediction_df,accuracy


def linear(bist100):
    #bist100 = pd.read_csv(data)
    bist100.rename(columns={"Date":"date","Open":"open","High":"high","Low":"low","Close":"close"}, inplace= True)
    bist100['date'] = pd.to_datetime(bist100.date)
    bist100.sort_values(by='date', inplace=True)
    bist100["close"] = bist100["close"].replace({'\$': ''}, regex=True)
    bist100["close"] = bist100["close"].apply(pd.to_numeric)
    closedf = bist100[['date','close']]
    print("start")
    close_stock = closedf.copy()
    del closedf['date']
    scaler=MinMaxScaler(feature_range=(0,1))
    closedf=scaler.fit_transform(np.array(closedf).reshape(-1,1))

    training_size=int(len(closedf)*0.65)
    test_size=len(closedf)-training_size
    train_data,test_data=closedf[0:training_size,:],closedf[training_size:len(closedf),:1]


    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    time_step = 15
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
    model = LinearRegression()


    X_train_linear = X_train.reshape(X_train.shape[0], X_train.shape[1])
    X_test_linear = X_test.reshape(X_test.shape[0], X_test.shape[1])

    model.fit(X_train_linear, y_train)


    y_pred = model.predict(X_test_linear)

    accuracy= r2_score(y_test, y_pred)
    #print(accuracy)

    train_predict = model.predict(X_train_linear)
    test_predict = model.predict(X_test_linear)

    train_predict = scaler.inverse_transform(train_predict.reshape(-1,1))
    test_predict = scaler.inverse_transform(test_predict.reshape(-1,1))
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
    original_ytest = scaler.inverse_transform(y_test.reshape(-1,1))
    look_back=time_step

    trainPredictPlot = np.empty_like(closedf)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

    testPredictPlot = np.empty_like(closedf)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(closedf)-1, :] = test_predict

    names = cycle(['Original close price','Train predicted close price','Test predicted close price'])

    plotdf = pd.DataFrame({'date': close_stock['date'],
                           'original_close': close_stock['close'],
                          'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                          'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})

    fig = px.line(plotdf,x=plotdf['date'], y=[plotdf['original_close'],plotdf['train_predicted_close'],
                                              plotdf['test_predicted_close']],
                  labels={'value':'Stock price','date': 'Date'})
    fig.update_layout(title_text='Comparison between original close price vs predicted close price',
                      plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
    fig.for_each_trace(lambda t:  t.update(name = next(names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.write_image("static/linear/linear_train.jpg")
    # Assuming you've already trained your linear regression model and it's stored in 'model'

    # Reshape test data for prediction
    x_input = test_data[len(test_data) - time_step:].reshape(1, -1)

    # Initialize temp_input as a list
    temp_input = list(x_input[0])

    lst_output = []
    pred_days = 7

    for i in range(pred_days):
        if len(temp_input) > time_step:
            # Extract the most recent 'time_step' data points
            x_input = np.array(temp_input[1:])
            # Reshape for prediction
            x_input = x_input.reshape(1, -1)
            # Predict the next value
            yhat = model.predict(x_input)
            # Append the prediction to temp_input
            temp_input.extend(yhat.tolist())
            # Move the window by one step
            temp_input = temp_input[1:]
            # Append the prediction to the output list
            lst_output.extend(yhat.tolist())
        else:
            # Reshape for prediction
            x_input = np.array(temp_input).reshape(1, -1)
            # Predict the next value
            yhat = model.predict(x_input)
            # Append the prediction to temp_input
            temp_input.extend(yhat.tolist())
            # Append the prediction to the output list
            lst_output.extend(yhat.tolist())
    last_days=np.arange(1,time_step+1)
    day_pred=np.arange(time_step+1,time_step+pred_days+1)
    temp_mat = np.empty((len(last_days)+pred_days+1,1))
    temp_mat[:] = np.nan
    temp_mat = temp_mat.reshape(1,-1).tolist()[0]

    last_original_days_value = temp_mat
    next_predicted_days_value = temp_mat

    last_original_days_value[0:time_step+1] = scaler.inverse_transform(closedf[len(closedf)-time_step:]).reshape(1,-1).tolist()[0]
    next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

    new_pred_plot = pd.DataFrame({
        'last_original_days_value':last_original_days_value,
        'next_predicted_days_value':next_predicted_days_value
    })

    names = cycle(['Last 15 days close price','Predicted next 7 days close price'])

    fig = px.line(new_pred_plot,x=new_pred_plot.index, y=[new_pred_plot['last_original_days_value'],
                                                          new_pred_plot['next_predicted_days_value']],
                  labels={'value': 'Stock price','index': 'Timestamp'})
    fig.update_layout(title_text=' next 7 days',
                      plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')

    fig.for_each_trace(lambda t:  t.update(name = next(names)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.write_image("static/linear/future_linear.jpg")
    lstmdf=closedf.tolist()
    lstmdf.extend((np.array(lst_output).reshape(-1,1)).tolist())
    lstmdf=scaler.inverse_transform(lstmdf).reshape(1,-1).tolist()[0]

    names = cycle(['Close price'])

    fig = px.line(lstmdf,labels={'value': 'Stock price','index': 'Timestamp'})
    fig.update_layout(title_text='Plotting whole closing stock price with prediction',
                      plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Stock')

    fig.for_each_trace(lambda t:  t.update(name = next(names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.write_image("static/linear/final_linear.jpg")
    predicted_values = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1)).reshape(1, -1).tolist()[0]
    combined_values = list(zip( predicted_values))
    prediction_df = pd.DataFrame(combined_values, columns=[ 'Predicted Close Price'])
    return prediction_df,accuracy




    

def arima(bist100):
    # Load the dataset
    #bist100 = pd.read_csv(data)

    # Rename columns
    bist100.rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close"}, inplace=True)

    # Convert date column to datetime
    bist100['date'] = pd.to_datetime(bist100['date'])

    # Sort the dataframe by date
    bist100.sort_values(by='date', inplace=True)

    # Select the 'close' column for modeling
    closedf = bist100[['date', 'close']]

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    closedf['close'] = scaler.fit_transform(closedf['close'].values.reshape(-1, 1))

    # Split the dataset into train and test sets
    training_size = int(len(closedf) * 0.65)
    train_data, test_data = closedf.iloc[0:training_size], closedf.iloc[training_size:len(closedf)]

    # Create ARIMA dataset
    train_arima = train_data['close'].values
    test_arima = test_data['close'].values

    # Fit ARIMA model
    model = ARIMA(train_arima, order=(5, 1, 0))  # Example order, you may need to tune this
    model_fit = model.fit()


    predictions = model_fit.forecast(steps=len(test_arima))
    accuracy = r2_score(test_arima, predictions)



    look_back=15

    train_predict = model_fit.predict(start=1, end=len(train_arima))
    test_predict = model_fit.forecast(steps=len(test_arima))

    # Truncate train and test predictions to match the original data length
    trainPredictPlot = np.empty((len(closedf), 1))
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, 0] = train_predict

    # Adjust the length of testPredictPlot
    testPredictPlot = np.empty((len(closedf), 1))
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict) + (-1) + 1:len(closedf) + len(test_predict) + 1, 0] = test_predict


    names = cycle(['Original close price','Train predicted close price','Test predicted close price'])


    # Create DataFrame for plotting
    plotdf = pd.DataFrame({'date': closedf['date'],
                           'original_close': closedf['close'],
                           'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                           'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})
    fig = px.line(plotdf,x=plotdf['date'], y=[plotdf['original_close'],plotdf['train_predicted_close'],
                                              plotdf['test_predicted_close']],
                  labels={'value':'Stock price','date': 'Date'})
    fig.update_layout(title_text='Comparison between original close price vs predicted close price',
                      plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
    fig.for_each_trace(lambda t:  t.update(name = next(names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.write_image("static/arima/arima_train.jpg")
    time_step=15

    x_input = test_arima[len(test_arima) - time_step:]

    # Initialize temp_input as a list
    temp_input = list(x_input)

    lst_output = []
    pred_days = 7
    for i in range(pred_days):
        if len(temp_input) > time_step:
            # Extract the most recent 'time_step' data points
            x_input = np.array(temp_input[1:])
            # Reshape for prediction
            x_input = x_input.reshape(1, -1)
            # Forecast the next value
            yhat = model_fit.forecast(steps=1)
            # Append the forecast to temp_input
            temp_input.extend(yhat.tolist())
            # Move the window by one step
            temp_input = temp_input[1:]
            # Append the forecast to the output list
            lst_output.extend(yhat.tolist())
        else:
            # Reshape for prediction
            x_input = np.array(temp_input).reshape(1, -1)
            # Forecast the next value
            yhat = model_fit.forecast(steps=1)
            # Append the forecast to temp_input
            temp_input.extend(yhat.tolist())
            # Append the forecast to the output list
            lst_output.extend(yhat.tolist())

    last_days=np.arange(1,time_step+1)
    day_pred=np.arange(time_step+1,time_step+pred_days+1)
    # Initialize temp_mat with the correct length
    temp_mat = np.empty((len(last_days) + pred_days + 1, 1))
    temp_mat[:] = np.nan
    temp_mat = temp_mat.reshape(1, -1).tolist()[0]

    # Assign temp_mat to last_original_days_value and next_predicted_days_value
    last_original_days_value = temp_mat
    next_predicted_days_value = temp_mat

    # Update last_original_days_value and next_predicted_days_value with correct values
    last_original_days_value[0:time_step+1] = scaler.inverse_transform(closedf[len(closedf) - time_step:]['close'].values.reshape(-1, 1)).reshape(1, -1).tolist()[0]
    next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1)).reshape(1, -1).tolist()[0]

    # Create DataFrame for plotting
    new_pred_plot = pd.DataFrame({
        'last_original_days_value': last_original_days_value,
        'next_predicted_days_value': next_predicted_days_value
    })


    names = cycle(['Last 15 days close price','Predicted next 7 days close price'])

    fig = px.line(new_pred_plot,x=new_pred_plot.index, y=[new_pred_plot['last_original_days_value'],
                                                          new_pred_plot['next_predicted_days_value']],
                  labels={'value': 'Stock price','index': 'Timestamp'})
    
    fig.update_layout(title_text=' next 7 days',
                      plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')

    fig.for_each_trace(lambda t:  t.update(name = next(names)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.write_image("static/arima/future_arima.jpg")
    # Convert DataFrame columns to lists
    lstmdf = closedf['close'].tolist()  # Assuming 'close' is the column containing stock prices

    # Extend lstmdf with the predicted values
    lstmdf.extend(np.array(lst_output).flatten().tolist())

    # Reshape lstmdf to a list of lists, each representing a single data point
    lstmdf_2d = [[x] for x in lstmdf]

    # Inverse transform the scaled data
    lstmdf_scaled = scaler.inverse_transform(lstmdf_2d).flatten().tolist()

    # Plot the data
    names = cycle(['Close price'])

    fig = px.line(lstmdf_scaled, labels={'value': 'Stock price', 'index': 'Timestamp'})
    fig.update_layout(title_text='Plotting whole closing stock price with prediction',
                      plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Stock')

    fig.for_each_trace(lambda t: t.update(name=next(names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.write_image("static/arima/final_arima.jpg")
    # Convert the predicted values to a list
    predicted_values = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1)).reshape(1, -1).tolist()[0]

    # Select only the last 10 days of predicted values
    last_7_predicted_values = predicted_values[-7:]

    # Create a DataFrame for the last 10 predicted values
    prediction_df = pd.DataFrame(last_7_predicted_values, columns=['Predicted Close Price'])

    # Print the DataFrame
    return prediction_df,accuracy

database = "new.db"
conn = sqlite3.connect(database)
cur = conn.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS register (username TEXT, usermail TEXT, password INT)")
conn.commit()

app = Flask(__name__, static_url_path='/static', static_folder='static')
import os
app.secret_key = '123'




@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get("username")
    usermail = data.get("usermail")
    password = data.get("password")
    conn = sqlite3.connect(database)
    cur = conn.cursor()
    cur.execute("INSERT INTO register (username, usermail, password) VALUES (?, ?, ?)",
                (username, usermail, password))
    conn.commit()
    conn.close()
    
    return jsonify({"success": True, "message": "Registration successful"})


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()  
    usermail = data.get("usermail")
    password = data.get("password")
    
    conn = sqlite3.connect(database)
    cur = conn.cursor()
    cur.execute("SELECT * FROM register WHERE usermail=? AND password=?", (usermail, password))
    user = cur.fetchone()
    conn.close()

    if user:
        session['usermail'] = usermail
        return jsonify({"success": True, "redirect_url": "/data1"})
    else:
        return jsonify({"success": False, "message": "Invalid email or password"}), 400  

    
@app.route('/data1', methods=['GET','POST'])
def data1():
    return render_template("data.html")

lstm_predict1=[]
linear_predict1=[]
arima_predict1=[]
accuracies = []
@app.route('/data', methods=['POST'])
def data():
    lstm_predict1.clear()
    linear_predict1.clear()
    arima_predict1.clear()
    accuracies.clear()
    data = request.files['csv_file']
    bist100 = pd.read_csv(data)
    lstm_predict, accuracy1 = lstm(bist100)
    lstm_predict1.append(lstm_predict)
    
    accuracies.append(('LSTM', accuracy1))
    linear_predict, accuracy2 = linear(bist100)
    linear_predict1.append(linear_predict)
    
    accuracies.append(('Linear', accuracy2))
    arima_predict, accuracy3 = arima(bist100)
    arima_predict1.append(arima_predict)
    accuracies.append(('ARIMA', accuracy3))
    return render_template("traning.html")


    
@app.route('/result', methods=['GET', 'POST'])
def result():
    max_accuracy_algo = max(accuracies, key=lambda x: x[1])
    algorithm, accuracy = max_accuracy_algo[0], max_accuracy_algo[1]

    if algorithm == 'LSTM':
        predictions_df = lstm_predict1[-1]
    elif algorithm == 'Linear':
        predictions_df = linear_predict1[-1]
    elif algorithm == 'ARIMA':
        predictions_df = arima_predict1[-1]

    prediction_values = predictions_df['Predicted Close Price'].tolist()


    labeled_predictions = {str(i + 1): round(val, 2) for i, val in enumerate(prediction_values)}

    return render_template("result.html",
                           accuracy=round(accuracy, 4),
                           algorithm=algorithm,
                           predictions=labeled_predictions)
        


from flask_cors import CORS
import yfinance as yf
from sklearn.linear_model import LinearRegression
from datetime import datetime

CORS(app) 
@app.route('/')
def index():
    return render_template("dashboard.html")

def fetch_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="60d")  
        
        if hist.empty or len(hist) < 10:
            return None

        hist = hist[['Close']].reset_index()
        hist['Day'] = (hist['Date'] - hist['Date'].min()).dt.days  

        X = hist[['Day']]
        y = hist['Close']
        model = LinearRegression()
        model.fit(X, y)

    
        next_day = hist['Day'].max() + 1
        predicted_price = model.predict([[next_day]])[0]

    
        chart_history = hist.tail(5).copy()
        chart_data = [
            {"date": row['Date'].strftime('%Y-%m-%d'), "price": round(row['Close'], 2)}
            for _, row in chart_history.iterrows()
        ]


        current_price = y.iloc[-1]
        info = stock.info
        return {
            "symbol": symbol.upper(),
            "currentPrice": round(current_price, 2),
            "marketCap": info.get("marketCap", 0),
            "volume": info.get("volume", 0),
            "predictedPrice": round(predicted_price, 2),
            "history": chart_data
        }

    except Exception as e:
        print("Error fetching or predicting:", e)
        return None



@app.route('/api/stock')
def stock():
    symbol = request.args.get('symbol', '').upper()  
    if not symbol:
        return jsonify({"error": "Stock symbol required"}), 400 

    data = fetch_stock_data(symbol)
    if data:
        return jsonify(data)  
    else:
        return jsonify({"error": "Could not fetch data"}), 500  


@app.route('/api/index')
def index_data():
    data = fetch_index_data()
    if data:
        return jsonify(data)  
    else:
        return jsonify({"error": "Could not fetch index data"}), 500  

if __name__ == '__main__':
    app.run(debug=False,port=600)
