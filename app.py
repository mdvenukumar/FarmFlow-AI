# app.py

from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
import time 

app = Flask(__name__)

# Load your dataset and train the model
file_path = 'dataset.csv'  # Adjust the path accordingly
df = pd.read_csv(file_path)

X = df[['nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Assuming 'label' is the target variable
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose a model (Random Forest in this case)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

def predict_label(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    new_data = pd.DataFrame([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]],
                            columns=['nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall'])
    
    label_prediction = model.predict(new_data)[0]
    
    # Find more neighbors to increase the chances of having different crops
    neighbors = NearestNeighbors(n_neighbors=10).fit(X)
    distances, indices = neighbors.kneighbors(new_data)
    
    # Get the labels of the neighbors excluding the predicted label
    nearest_labels = y.iloc[indices[0]]
    
    # Filter out the predicted label and get unique crops
    other_nearest_crops_data = df.loc[indices[0]][df['label'] != label_prediction][['label']].drop_duplicates().values
    other_nearest_crops = [crop[0] for crop in other_nearest_crops_data][:2]

    # Retrieve the image URL for the predicted label
    predicted_label_image = get_image_url(label_prediction)

    # Retrieve the image URLs for other nearest crops
    other_nearest_crops_images = [get_image_url(crop) for crop in other_nearest_crops]

    return label_prediction, predicted_label_image, other_nearest_crops, other_nearest_crops_images

def get_image_url(crop_label):
    image_urls_df = pd.read_csv('images.csv')  # Adjust the file path accordingly
    return image_urls_df[image_urls_df['label'] == crop_label]['image_url'].iloc[0]

@app.route('/')
def landing():
    plants_df = pd.read_csv('images.csv')
    plants_data = []
    for _, row in plants_df.iterrows():
        plant = {'label': row['label'], 'image_url': row['image_url']}
        plants_data.append(plant)

    return render_template('landing.html', plants=plants_data)

@app.route('/predict', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        nitrogen = float(request.form['nitrogen'])
        phosphorus = float(request.form['phosphorus'])
        potassium = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Use the predict_label function
        predicted_label, predicted_label_image, other_nearest_crops, other_nearest_crops_images = predict_label(
            nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)

        if not other_nearest_crops:
            no_other_crops_message = "No other crops suit your soil that are available in the training data."
            return render_template('result.html', predicted_label=predicted_label, 
                                   predicted_label_image=predicted_label_image,
                                   no_other_crops_message=no_other_crops_message)

        return render_template('result.html', predicted_label=predicted_label,
                               predicted_label_image=predicted_label_image,
                               other_nearest_crops=other_nearest_crops,
                               other_nearest_crops_images=other_nearest_crops_images)

    # If it's a GET request, you might want to handle it differently or redirect
    return redirect(url_for('landing'))

if __name__ == '__main__':
    app.run()
