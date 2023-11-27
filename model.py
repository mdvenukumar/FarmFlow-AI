import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors

# Load your dataset from the CSV file
file_path = 'dataset.csv'  # Adjust the path accordingly
df = pd.read_csv(file_path)

# Assuming 'label' is the target variable
X = df[['nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose a model (Random Forest in this case)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f'Model Accuracy: {accuracy}')

# Now, you can use the trained model to predict labels for new input values
def predict_label(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    new_data = pd.DataFrame([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]],
                            columns=['nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall'])
    
    label_prediction = model.predict(new_data)[0]
    
    # Find more neighbors to increase the chances of having different crops
    neighbors = NearestNeighbors(n_neighbors=10).fit(X)
    distances, indices = neighbors.kneighbors(new_data)
    
    # Get the labels of the neighbors excluding the predicted label
    nearest_labels = y.iloc[indices[0]]
    other_nearest_crops = nearest_labels[nearest_labels != label_prediction].values[:2]
    
    return label_prediction, other_nearest_crops

# Example usage:
user_input = (66, 78, 45, 27.8,75.3, 7.89, 1271.6)
predicted_label, other_nearest_crops = predict_label(*user_input)
unique_labels = df['label'].unique()
print(f'Predicted Label: {predicted_label}')
'''


label_counts = df['label'].value_counts()
Enter your location: East Godavari
Enter nitrogen level: 66
Enter phosphorus level: 78
Enter potassium level: 45
Enter pH level: 7.89
Location: East Godavari
Predicted Crop: banana

print(f'Unique Labels: {unique_labels}')
print('\nLabel Counts:')
print(label_counts)
'''