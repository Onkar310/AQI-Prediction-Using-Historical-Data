


from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the selected date from the form
    selected_date = request.form['date']

    # Convert the selected date to datetime format
    date_to_predict = pd.to_datetime(selected_date, format='%Y-%m-%d')

    # Create a DataFrame with the hourly timestamps for prediction
    hours_to_predict = pd.date_range(date_to_predict + pd.Timedelta(hours=10), date_to_predict + pd.Timedelta(hours=18), freq='H')
    future = pd.DataFrame({'ds': hours_to_predict})

    # Make prediction
    forecast = model.predict(future)

    # Get the predicted AQI values for the specified hours
    hourly_aqi = forecast[['ds', 'yhat']].rename(columns={'ds': 'date', 'yhat': 'aqi'})

    # Convert hourly AQI values to array format
    hourly_aqi_array = hourly_aqi.to_dict(orient='records')

    return jsonify(hourly_aqi_array)

if __name__ == '__main__':
    app.run(debug=True)
