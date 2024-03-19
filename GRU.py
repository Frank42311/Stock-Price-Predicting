# Standard library imports
import os
import time
from datetime import datetime

# Data handling
import pandas as pd
import numpy as np

# Machine Learning and Neural Networks
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import GRU, Dense, Concatenate, Reshape
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Utilities
import pickle
from joblib import load

# Visualization
import matplotlib.pyplot as plt

#################################################################################################################################
# 
# Split dataset: train, val, test
# 
#################################################################################################################################

# Define the directory path where the stock indicators are stored.
stocks_dir_path = r'data//us_stocks_indicators//'
window = 3  # The window size for how many rows to skip in the data.

total_size = 0  # Initialize the total size variable to keep track of the total number of data points.

# Loop through each file in the directory to calculate the total data points available.
for file_name in os.listdir(stocks_dir_path):
    file_path = os.path.join(stocks_dir_path, file_name)
    df = pd.read_csv(file_path)
    total_size += max(1, (len(df) - 60) // window)  # Calculate data points per file.

# Define the ratio of data to be split into train, validation, and test datasets.
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Ensure the sum of ratios equals 1.
assert train_ratio + val_ratio + test_ratio == 1

# Calculate the sizes of each dataset based on the total size and ratios.
train_size = int(total_size * train_ratio)
val_size = int(total_size * val_ratio)
test_size = total_size - train_size - val_size

# Preallocate numpy arrays for the datasets.
x_train = np.zeros((train_size, 60, 19))
y_train = np.zeros(train_size)
x_val = np.zeros((val_size, 60, 19))
y_val = np.zeros(val_size)
x_test = np.zeros((test_size, 60, 19))
y_test = np.zeros(test_size)

train_index, val_index, test_index = 0, 0, 0  # Initialize dataset indices.
group_index = 0  # Keep track of the current group index for splitting data.
total_time = 0  # Initialize a variable to track the total time taken for processing.

# Loop through each file in the directory again to populate the datasets.
for file_name in os.listdir(stocks_dir_path):
    file_path = os.path.join(stocks_dir_path, file_name)
    df = pd.read_csv(file_path)
    
    rows, cols = df.shape
    start_time = time.time()  # Record the start time for processing this file.

    # Loop through the data frame with the defined window size.
    for start_row in range(0, rows - 60, window):
        x_group = df.iloc[start_row:start_row+60, :].values  # Extract a window of data.
        y_group = df.iloc[start_row+60, 2]  # Assuming the 'Close' column is the third column.

        # Allocate the data to the appropriate dataset based on the current index.
        if group_index < train_size:
            x_train[train_index] = x_group
            y_train[train_index] = y_group
            train_index += 1
        elif group_index < train_size + val_size:
            x_val[val_index] = x_group
            y_val[val_index] = y_group
            val_index += 1
        else:
            x_test[test_index] = x_group
            y_test[test_index] = y_group
            test_index += 1

        group_index += 1

    # Update the total time taken with the time for processing this file.
    total_time += time.time() - start_time

# Print the total time taken for the entire operation.
print(f"Total time taken: {total_time:.2f} seconds")

# Split the datasets into smaller subsets based on the columns for further processing or model training.
x1_train = x_train[:, :, 0:1]    # First column.
x2_train = x_train[:, :, 1:5]    # Next four columns.
x3_train = x_train[:, :, 5:6]    # Next column.
x4_train = x_train[:, :, 6:14]   # Columns up to the 14th (indicators).
x5_train = x_train[:, :, 14:]    # Remaining columns (cdl patterns).

# Repeat the process for the validation and test datasets.
x1_val = x_val[:, :, 0:1]
x2_val = x_val[:, :, 1:5]
x3_val = x_val[:, :, 5:6]
x4_val = x_val[:, :, 6:14]
x5_val = x_val[:, :, 14:]

x1_test = x_test[:, :, 0:1]
x2_test = x_test[:, :, 1:5]
x3_test = x_test[:, :, 5:6]
x4_test = x_test[:, :, 6:14]
x5_test = x_test[:, :, 14:]

#################################################################################################################################
# 
# Model Building
# 
#################################################################################################################################

# Set the GRU layer's parameters
units = 64
activation = 'tanh'
recurrent_activation = 'sigmoid'

# Define the shapes of the input layers based on the preprocessed datasets
input_shape1 = x1_train.shape[1:]
input_shape2 = x2_train.shape[1:]
input_shape3 = x3_train.shape[1:]
input_shape4 = x4_train.shape[1:]
input_shape5 = x5_train.shape[1:]

# Initialize input layers for each group of features
input_group1 = Input(shape=input_shape1)
input_group2 = Input(shape=input_shape2)
input_group3 = Input(shape=input_shape3)
input_group4 = Input(shape=input_shape4)
input_group5 = Input(shape=input_shape5)

# Define GRU layers for groups 1, 3, 4, and 5
gru_group1 = GRU(30, activation='tanh', recurrent_activation='sigmoid')(input_group1)
gru_group3 = GRU(30, activation='tanh', recurrent_activation='sigmoid')(input_group3)
gru_group4 = GRU(30, activation='tanh', recurrent_activation='sigmoid')(input_group4)
gru_group5 = GRU(30, activation='tanh', recurrent_activation='sigmoid')(input_group5)

# Concatenate the outputs of these GRU layers
merged_abcd = Concatenate()([gru_group1, gru_group3, gru_group4, gru_group5])
dense_abcd_1 = Dense(8, activation='relu')(merged_abcd)
dense_abcd_2 = Dense(16, activation='relu')(dense_abcd_1)

# Define a GRU layer for group 2
gru_group2 = GRU(60, activation='tanh', recurrent_activation='sigmoid')(input_group2)

# Concatenate the result with the previously concatenated and processed outputs
merged_ef = Concatenate()([dense_abcd_2, gru_group2])

# Add a fully connected layer
dense_ef = Dense(64, activation='relu')(merged_ef)

# Reshape the output for compatibility with the next GRU layer
reshaped_ef = Reshape((1, 64))(dense_ef)

# Define another GRU layer
gru_ef = GRU(60, activation='tanh', recurrent_activation='sigmoid')(reshaped_ef)

# Final dense layers to output the prediction
dense_final_1 = Dense(16, activation='relu')(gru_ef)
dense_final_2 = Dense(1, activation='relu')(dense_final_1)  # Adjust based on your task

# Construct the model
model = Model(inputs=[input_group1, input_group2, input_group3, input_group4, input_group5], outputs=dense_final_2)

# Print the model architecture
model.summary()

#################################################################################################################################
# 
# Training
# 
#################################################################################################################################

# Callbacks for adjusting learning rate and early stopping.
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)
callbacks = [reduce_lr, early_stop]

# Compile the model with Adam optimizer and mean squared error loss function.
optimizer = Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model on the training dataset and validate on the validation dataset.
history = model.fit([x1_train, x2_train, x3_train, x4_train, x5_train], y_train,
                    validation_data=([x1_val, x2_val, x3_val, x4_val, x5_val], y_val),
                    epochs=100, batch_size=100, callbacks=callbacks)

# Print training history
print("Training History:")
print("Epochs: ", history.epoch)
print("Train Loss: ", history.history['loss'])
print("Validation Loss: ", history.history['val_loss'])

# Visualize the training and validation loss over epochs.
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

#################################################################################################################################
# 
# Testing Preparing and Predicting
# 
#################################################################################################################################

# Load test data and predict
y_test_pred = model.predict([x1_test, x2_test, x3_test, x4_test, x5_test])

# Calculate MSE for the test dataset
mse_test = mean_squared_error(y_test, y_test_pred)
print(f"Test dataset MSE: {mse_test}")

# Load data for a specific stock (106.ARR.csv)
ARR_path = r'data//us_stocks_indicators_arr//'
arr_file_path = os.path.join(ARR_path, '106.ARR.csv')
arr_df = pd.read_csv(arr_file_path)

# Split the data based on a window size of 1
x_arr = np.zeros((len(arr_df) - 60, 60, 19))
y_arr = np.zeros(len(arr_df) - 60)

for start_row in range(0, len(arr_df) - 60):
    x_arr[start_row] = arr_df.iloc[start_row:start_row+60, :].values  # Extracting 60 rows of data for each sample
    y_arr[start_row] = arr_df.iloc[start_row+60, 2]  # Extracting the 'Close' price as the target variable

# Split the prepared data into five different input groups for the model
x1_arr = x_arr[:, :, 0:1]    # First column
x2_arr = x_arr[:, :, 1:5]    # Next four columns
x3_arr = x_arr[:, :, 5:6]    # The subsequent column
x4_arr = x_arr[:, :, 6:14]   # Columns from the sixth to the fourteenth (indicators)
x5_arr = x_arr[:, :, 14:]    # The remaining columns (cdl patterns)

# Predict with the model
y_arr_pred = model.predict([x1_arr, x2_arr, x3_arr, x4_arr, x5_arr])

# Calculate the Mean Squared Error (MSE) for predictions
mse_arr = mean_squared_error(y_arr, y_arr_pred)
print(f"MSE for 106.ARR.csv: {mse_arr}")

# Combine the actual and predicted values into a DataFrame
df = pd.DataFrame({
    'y_arr': y_arr,
    'y_arr_pred': y_arr_pred.reshape(-1)  # Ensure y_arr_pred is a 1D array for consistent DataFrame structure
})

# Exclude samples where the actual value is zero to avoid division by zero in error calculation
df_filtered = df[df['y_arr'] != 0]

# Recalculate the Mean Percentage Error on the filtered dataset
mean_percentage_error_filtered = np.mean(np.abs((df_filtered['y_arr'] - df_filtered['y_arr_pred']) / df_filtered['y_arr'])) * 100
print(f"Filtered Mean Percentage Error: {mean_percentage_error_filtered}%")

#################################################################################################################################
# 
# Descaling
# 
#################################################################################################################################

# Load the price scaler to transform the target variable back to its original scale
price_scaler_path = os.path.join(ARR_path, '106.ARR.csv_price.pkl')
price_scaler = load(price_scaler_path)

# Inverse transform the actual and predicted values using the price scaler
# Duplicate y_arr into four columns to match the scaler's expected input format
y_arr_temp = np.repeat(y_arr.reshape(-1, 1), 4, axis=1)
y_arr_real_temp = price_scaler.inverse_transform(y_arr_temp)  # Inverse transform to original price scale
y_arr_real = y_arr_real_temp[:, 0].reshape(-1, 1)  # Extract the first column as the actual prices

# Repeat the inverse transformation process for the predicted values
y_arr_pred_temp = np.repeat(y_arr_pred.reshape(-1, 1), 4, axis=1)
y_arr_pred_real_temp = price_scaler.inverse_transform(y_arr_pred_temp)
y_arr_pred_real = y_arr_pred_real_temp[:, 0].reshape(-1, 1)

# Load the datetime scaler to convert the date column back to datetime format
datetime_scaler_path = os.path.join(ARR_path, 'datetime_scaler.save')
datetime_scaler = load(datetime_scaler_path)

# Inverse transform the date column from the dataset
date_column = arr_df['date'].values[-len(y_arr):]  # Select the appropriate timestamp column from the DataFrame
date_column_timestamps = datetime_scaler.inverse_transform(date_column.reshape(-1, 1))

# Convert Unix timestamps back to datetime objects
date_column_real = [pd.Timestamp(ts[0], unit='s') for ts in date_column_timestamps]

# Combine dates with their corresponding real and predicted prices into a DataFrame for visualization
data = pd.DataFrame({
    'Date': date_column_real,
    'Real Price': y_arr_real.flatten(),
    'Predicted Price': y_arr_pred_real.flatten()
})

# Filter the data to a specific date range for analysis
start_date = pd.to_datetime("2020-04-01")
end_date = pd.to_datetime("2022-12-01")
filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

# Plot the real versus predicted prices for the specified date range
plt.figure(figsize=(15, 6))
plt.plot(filtered_data['Date'], filtered_data['Real Price'], label='Real Price')
plt.plot(filtered_data['Date'], filtered_data['Predicted Price'], label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Price Comparison for ARR from 2020 to 2022')
plt.legend()
plt.show()