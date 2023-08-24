import netCDF4 as nc
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Load the NetCDF Data
data_path = "D:/data/GRACEDADM_CLSM025GL_7D.A20230327.030.nc4"
data = nc.Dataset(data_path)

# Extract the required variable data
lat = data.variables["lat"][:]
lon = data.variables["lon"][:]
time = data.variables["time"][:]
gws_inst = data.variables["gws_inst"][:]
rtzsm_inst = data.variables["rtzsm_inst"][:]
sfsm_inst = data.variables["sfsm_inst"][:]

# Reshape the variables as needed (assuming they are 2D arrays)
lat_2d, lon_2d = np.meshgrid(lat, lon)
time_reshaped = np.repeat(time, lat.shape[0] * lon.shape[0])

# Combine the variables into a feature matrix
features = np.column_stack((lat_2d.flatten(), lon_2d.flatten(), time_reshaped,
                            rtzsm_inst.flatten(), sfsm_inst.flatten()))

# Use "gws_inst" as the target variable
target_variable = gws_inst.flatten()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target_variable, test_size=0.2, random_state=42)

# Step 3: Train the Random Forest Model
random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

# Step 4: Make Predictions
y_pred = random_forest.predict(X_test)

# Visualize the Actual and Predicted Values with Different Colors
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.7, label='Predicted')
plt.scatter(y_test, y_test, color='green', alpha=0.5, label='Actual')
plt.title('Predicted vs Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()

# Calculate Mean Squared Error as an example evaluation metric
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
