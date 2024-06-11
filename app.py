from flask import Flask, request, redirect, url_for, render_template, send_file
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CONTACTS_FILE'] = os.path.join(app.config['UPLOAD_FOLDER'], 'contacts.csv')

# Load the trained RandomForest model
with open('model\model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def dashboard():
    return render_template('1dasboard.html')

@app.route('/feature1')
def feature1():
    return render_template('2feature1.html')

@app.route('/feature2')
def feature2():
    return render_template('3feature2.html')

@app.route('/aboutus')
def aboutus():
    return render_template('5aboutus.html')

@app.route('/contactus', methods=['GET', 'POST'])
def contactus():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        message = request.form['message']
        
        contact_data = pd.DataFrame([{'Name': name, 'Email': email, 'Phone': phone, 'Message': message}])
        if not os.path.isfile(app.config['CONTACTS_FILE']):
            contact_data.to_csv(app.config['CONTACTS_FILE'], index=False)
        else:
            contact_data.to_csv(app.config['CONTACTS_FILE'], mode='a', header=False, index=False)
        
        return redirect(url_for('contactus'))
    return render_template('6contactus.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    global predictions, result_data, data 
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            data = pd.read_csv(filepath)
            
            # Expected columns
            expected_columns = [
                'Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 'Open', 'Promo',
                'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
                'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear',
                'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval'
            ]

            # Fill missing expected columns with default values
            for column in expected_columns:
                if column not in data.columns:
                    if column in ['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval']:
                        data[column] = '0'
                    else:
                        data[column] = 0

            # Keep the original columns
            original_data = data[['Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday']].copy()

            # Preprocess the data as done during training
            data['Month'] = pd.to_datetime(data['Date']).dt.month
            data['Year'] = pd.to_datetime(data['Date']).dt.year
            data['Day'] = pd.to_datetime(data['Date']).dt.day

            # Fill NA values and preprocess columns similar to your training set
            data.fillna(1, inplace=True)
            data['StateHoliday'] = data['StateHoliday'].astype('category').cat.codes
            data['Assortment'] = data['Assortment'].astype('category').cat.codes
            data['StoreType'] = data['StoreType'].astype('category').cat.codes
            data['PromoInterval'] = data['PromoInterval'].astype('category').cat.codes

            # Feature engineering
            data['CompetitionOpenSince'] = np.where(
                (data['CompetitionOpenSinceMonth'] == 0) & (data['CompetitionOpenSinceYear'] == 0),
                0,
                (data.Month - data.CompetitionOpenSinceMonth) + (12 * (data.Year - data.CompetitionOpenSinceYear))
            )
            data["is_holiday_state"] = data['StateHoliday'].map({0: 0, 1: 1, 2: 1, 3: 1})

            # Convert categorical variables into dummy/indicator variables
            data = pd.get_dummies(data, columns=["Assortment", "StoreType", "PromoInterval"])

            # Ensure all expected columns are present
            for col in model.feature_names_in_:
                if col not in data.columns:
                    data[col] = 0

            data = data[model.feature_names_in_]
            
            # Predict
            predictions = model.predict(data)
            predictions = np.exp(predictions)  # Convert from log scale if necessary

            # Combine predictions with original data
            original_data['Sales_Predicted'] = predictions
            result_data = original_data

            # Convert result_data to a list of dictionaries for template rendering
            result_list = result_data.to_dict(orient='records')

            # Generate visualizations
            generate_visualizations()

            return render_template('3feature2.html', predictions=result_list, plot_url1=url_for('plot_image1'), plot_url2=url_for('plot_image2'), plot_url3=url_for('plot_image3'))
        else:
            return 'Invalid file format. Only CSV files are allowed.'
    else:
        return render_template('3feature2.html', predictions=[])

def generate_visualizations():
    global img1, img2, img3
    # 1. Distribution of Predicted Sales
    plt.figure(figsize=(10, 6))
    sns.histplot(predictions, bins=50, kde=True)
    plt.title('Distribution of Predicted Sales')
    plt.xlabel('Predicted Sales')
    plt.ylabel('Frequency')
    plt.tight_layout()
    img1 = BytesIO()
    plt.savefig(img1, format='png')
    img1.seek(0)

    # 2. Top Selling Stores
    top_stores = result_data.groupby('Store')['Sales_Predicted'].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_stores.index, y=top_stores.values)
    plt.title('Top 10 Selling Stores')
    plt.xlabel('Store')
    plt.ylabel('Total Predicted Sales')
    plt.tight_layout()
    img2 = BytesIO()
    plt.savefig(img2, format='png')
    img2.seek(0)

    # 3. Feature Importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_features = data.columns[indices][:10]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices][:10], y=top_features)
    plt.title('Top 10 Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    img3 = BytesIO()
    plt.savefig(img3, format='png')
    img3.seek(0)

@app.route('/plot1.png')
def plot_image1():
    return send_file(img1, mimetype='image/png')

@app.route('/plot2.png')
def plot_image2():
    return send_file(img2, mimetype='image/png')

@app.route('/plot3.png')
def plot_image3():
    return send_file(img3, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
