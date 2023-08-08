from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the pre-trained LSTM model
loaded_LSTM_model = load_model('C:\\Users\\amitk\\OneDrive\\Desktop\\project_4 clean\\LSTM_model.h5')
# Compile the loaded model
loaded_LSTM_model.compile(optimizer='adam', loss='mean_squared_error')

# Initialize and fit the scaler
scaler = MinMaxScaler(feature_range=(0, 1))

def preprocess_input_data(input_data, scaler):
    num_features = input_data.shape[1]
    repeated_input = np.tile(input_data, (10, 1))  # Repeat the input to create sequences of length 10
    num_sequences = len(repeated_input) - 9  # Number of sequences of length 10

    if num_sequences <= 0:
        return None

    preprocessed_data = []
    for i in range(num_sequences):
        sequence = repeated_input[i:i + 10]  # Extract a sequence of length 10
        preprocessed_data.append(sequence)

    # Flatten the 3D array to a 2D array for scaling
    flattened_data = np.reshape(preprocessed_data, (-1, num_features))
    
    # Scale the flattened data using the provided scaler
    scaled_data = scaler.transform(flattened_data)

    # Reshape the scaled data back to the original 3D shape
    preprocessed_data = np.reshape(scaled_data, (num_sequences, 10, num_features))
    
    return preprocessed_data

@app.route('/', methods=['GET', 'POST'])
def dailypredict():
    if request.method == 'POST':
        # Extract the selected ticker value from the form
        ticker = request.form['ticker']

        # preprocess input data before making predictions
        price = float(request.form['Close'])
        volume = float(request.form['volume'])
        input_data = np.array([[price, volume, 1.0]])  # Shape: (1, 3)

        # Fit the scaler with the input data and preprocess it
        scaler.fit(input_data)
        preprocessed_input = preprocess_input_data(input_data, scaler)  # Shape: (1, 10, 3)

        # Now you can make predictions using the preprocessed input data
        prediction = loaded_LSTM_model.predict_on_batch(preprocessed_input)[0]
        

        # If the prediction is an array of arrays, get the first element as the result
        #prediction = prediction[0]

        return render_template('result1.html', ticker = ticker, prediction=prediction, price=price, volume=volume)

    return render_template('index2.html')

if __name__ == '__main__':
    app.run(debug=True)

