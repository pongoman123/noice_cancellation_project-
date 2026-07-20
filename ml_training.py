import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import m2cgen as m2c
import joblib

# 1. Load the LTspice generated dataset
print("Loading dataset...")
df = pd.read_csv('anc_training_dataset.csv')

# Define our Input (Frequency) and Output (Target Resistance)
X = df[['Frequency_Hz']]
y = df['Optimal_Rtune_Ohms']

# 2. Split the data (80% for training, 20% for testing the AI's accuracy)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and Train the Model
print("Training the Random Forest AI...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate the Model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Model Training Complete!")
print(f"Mean Squared Error: {mse:.2f}")

# 5. Visualize the AI's Learning
print("Generating performance graph...")
plt.figure(figsize=(10, 6))

# Plot the raw data points from LTspice (Blue dots)
plt.scatter(X, y, color='blue', label='LTspice Exact Data', alpha=0.4, s=20)

# Generate a smooth curve showing what the AI predicts across the whole spectrum (Red line)
X_smooth = np.linspace(X.min().iloc[0], X.max().iloc[0], 500).reshape(-1, 1)
# Convert to DataFrame to match the training feature names
X_smooth_df = pd.DataFrame(X_smooth, columns=['Frequency_Hz']) 
y_smooth = model.predict(X_smooth_df)

plt.plot(X_smooth, y_smooth, color='red', label='AI Prediction Curve', linewidth=2.5)

plt.title('Active Noise Cancellation: AI Tuning Model vs LTspice Physics')
plt.xlabel('Incoming Noise Frequency (Hz)')
plt.ylabel('Required Rtune Resistance (Ohms)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

print("Exporting model to C code...")
# Convert the trained Random Forest model to C
c_code = m2c.export_to_c(model)

# Save it as a header file that your microcontroller can read
with open("anc_model.h", "w") as f:
    f.write(c_code)

print("Success! Model saved as 'anc_model.h'.")
#saving the model for the use of software 
print("Saving the model for software use...")

# Save the trained model to a .pkl file
model_filename = 'anc_tuning_model.pkl'
joblib.dump(model, model_filename)

print(f"Success! Model securely saved as '{model_filename}'.")