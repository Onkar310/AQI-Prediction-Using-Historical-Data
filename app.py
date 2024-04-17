from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load the models and prediction function from the pickle file
with open('model.pkl', 'rb') as f:
    models = pickle.load(f)

# Function to predict hourly values for a given date
def predict_hourly_values(date_str):
    date = pd.to_datetime(date_str, format='%Y-%m-%d')
    hourly_predictions = []
    for pollutant, model in models.items():
        # Create a DataFrame with the hourly timestamps for prediction
        hours_to_predict = pd.date_range(date + pd.Timedelta(hours=10), date + pd.Timedelta(hours=18), freq='H')
        future = pd.DataFrame({'ds': hours_to_predict})

        # Make predictions
        forecast = model.predict(future)

        # Get the predicted values for the specified hours and convert to list format
        values = forecast['yhat'].tolist()

        # Append the pollutant name as the first element of the list
        values.insert(0, pollutant)

        # Append the list of predicted values to the hourly_predictions list
        hourly_predictions.append(values)

    return hourly_predictions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the date input from the form
    user_date = request.form['date']

    # Predict hourly values for the user input date
    predictions = predict_hourly_values(user_date)
    print(predictions)
    # Return the predictions as JSON response
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
