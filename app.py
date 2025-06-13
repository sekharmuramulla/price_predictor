from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

# Initialize the app
app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        crim = float(request.form['CRIM'])
        rm = float(request.form['RM'])
        dis = float(request.form['DIS'])
        tax = float(request.form['TAX'])
        ptratio = float(request.form['PTRATIO'])
        lstat = float(request.form['LSTAT'])

        # Put into DataFrame with correct column names (used during training)
        input_df = pd.DataFrame([{
            'crim': crim,
            'rm': rm,
            'dis': dis,
            'tax': tax,
            'ptratio': ptratio,
            'lstat': lstat
        }])

        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_text = f"Predicted House Price: ${prediction:,.2f}"

        return render_template('index.html', prediction_text=prediction_text)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
