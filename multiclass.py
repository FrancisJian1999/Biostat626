import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras import layers

file_path = 'training_data.txt'
df = pd.read_csv(file_path, sep='\s+', engine='python')
outpur_path = 'test_data.txt'
df1 = pd.read_csv(outpur_path, sep='\s+', engine='python')


# Change activity labels 7, 8, 9, 10, 11, and 12 to 7
df['activity'] = df['activity'].replace({8: 7, 9: 7, 10: 7, 11: 7, 12: 7})

# Separate features (X) and labels (y)
X = df.drop('activity', axis=1)
y = df['activity']

# Fit the scaler on the training data (X)
scaler = StandardScaler().fit(X)

# Scale the training data
X_scaled = scaler.transform(X)
df1_scaled = scaler.transform(df1)


# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# One-hot encode the labels
y_train_onehot = tf.keras.utils.to_categorical(y_train - 1)
y_test_onehot = tf.keras.utils.to_categorical(y_test - 1)

# Create a simple MLP model
model = tf.keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(y_train_onehot.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_onehot, batch_size=32, epochs=10, validation_split=0.1)

# Evaluate the model
y_pred_onehot = model.predict(X_test)
y_pred = y_pred_onehot.argmax(axis=1) + 1

# Print the performance metrics
print(classification_report(y_test, y_pred))
# Perform predictions on df1
df1_pred_onehot = model.predict(df1_scaled)
df1_pred = df1_pred_onehot.argmax(axis=1) + 1

# Save the predictions to a txt file
with open('predictions1.txt', 'w') as f:
    for prediction in df1_pred:
        f.write(f"{prediction}\n")
