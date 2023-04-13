import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras import layers

file_path = 'training_data.txt'
df = pd.read_csv(file_path, sep='\s+', engine='python')
output_path = 'test_data.txt'
df1 = pd.read_csv(output_path, sep='\s+', engine='python')

# Change activity labels
df['activity'] = df['activity'].replace({8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 7: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0})

# Separate features (X) and labels (y)
X = df.drop('activity', axis=1)
y = df['activity']

scaler = StandardScaler().fit(X)
# Scale the training data
X_scaled = scaler.transform(X)
df1_scaled = scaler.transform(df1)

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a simple MLP model for binary classification
model = tf.keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, batch_size=32, epochs=10, validation_split=0.1)

# Evaluate the model
y_pred = (model.predict(X_test_scaled) > 0.5).astype(int).flatten()

# Print the performance metrics
print(classification_report(y_test, y_pred))

# Perform predictions on df1
df1_pred = (model.predict(df1_scaled) > 0.5).astype(int).flatten()

# Save the predictions to a txt file
with open('predictions1.txt', 'w') as f:
    for prediction in df1_pred:
        f.write(f"{prediction}\n")
