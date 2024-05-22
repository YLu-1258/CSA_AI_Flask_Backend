import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import os

class LSTMClass:
    __script_dir = os.path.dirname(os.path.abspath(__file__))
    __prediction_destination = f"{__script_dir}/predictions/" + "{ticker}.csv" 
    __graph_destination = f"{__script_dir}/graphs/" + "{ticker}.png" 
    def __init__(self, ticker, stock_data):
        self.__ticker = ticker
        self.__stock_data = stock_data
        self.__scaler = MinMaxScaler(feature_range=(0,1))

    def plot_data(self):
        plt.figure(figsize=(16,8))
        plt.title(self.__ticker + ": Close Price History")
        plt.plot(self.__stock_data['Close'])    
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close price (US)', fontsize=18)
        plt.show()
    
    def scale_split_data(self):
        # Get close data
        close_data = self.__stock_data.filter(['Close'])
        close_dataset = close_data.values
        training_data_len = math.ceil(len(close_dataset) * .8)

        # Scale the data
        scaled_close_data = self.__scaler.fit_transform(close_dataset)
        training_data = scaled_close_data[0:training_data_len, :]
        test_data = scaled_close_data[training_data_len - 60:, :]

        # Split to Xtrain and Ytrain
        x_train = []
        y_train = []

        # Split to X_test and y_test
        x_test = []
        y_test = close_dataset[training_data_len:, :]

        for i in range(60, len(training_data)):
            x_train.append(training_data[i-60:i, 0])
            y_train.append(training_data[i, 0])

        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        data = {"x_train" : x_train, "y_train" : y_train, "x_test": x_test, "y_test": y_test, "training_data_len": training_data_len}
        return data
    
    def load_model(self):
        x_train, y_train = self.scale_split_data()['x_train'], self.scale_split_data()['y_train']
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        return model
    
    def train_and_predict_LSTM_model(self):
        data = self.scale_split_data()
        x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']
        model = self.load_model()
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, batch_size=1, epochs=1)
        predictions = model.predict(x_test)
        predictions = self.__scaler.inverse_transform(predictions)
        rmse_metric = np.sqrt( np.mean(predictions - y_test)**2)
        data['predictions'] = predictions
        data['rmse'] = rmse_metric
        return data

    def graph_predicted_data(self):
        data = self.export_predictions_to_csv()
        close_data = self.__stock_data.filter(['Close'])
        train = close_data[:data['training_data_len']]
        valid = close_data[data['training_data_len']:]
        valid['Predictions'] = data['predictions']
        plt.figure(figsize=(16,8))
        plt.title(self.__ticker + ": Model performance")  
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close price (US)', fontsize=18)
        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        plt.show()
        plt.savefig(self.__graph_destination.format(ticker=self.__ticker))
        
    def export_predictions_to_csv(self):
        data = self.train_and_predict_LSTM_model()
        valid = self.__stock_data[self.scale_split_data()['training_data_len']:].copy()
        valid['Predictions'] = data['predictions']
        predictions_df = valid[['Close', 'Predictions']].reset_index()
        predictions_df.columns = ['Date', 'Actual', 'Predictions']
        predictions_df.to_csv(self.__prediction_destination.format(ticker=self.__ticker), index=False)
        print(f"Predictions exported to {self.__prediction_destination.format(ticker=self.__ticker)}")
        return data