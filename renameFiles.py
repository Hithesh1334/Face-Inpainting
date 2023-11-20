# import os
# import cv2

# image_name = ""
# binary_path = ""
# binary = "" 
# base_filename, _ = os.path.splitext(image_name)
    
# output_filename = f"{base_filename}.jpg"  # Use the same name as the input image
# output_path_full = os.path.join(binary_path, output_filename)

# cv2.imwrite(output_path_full, binary * 255)  # Multiply by 255 to convert to 0-255 range
# print(f"Processed: {image_name} -> Saved as: {output_filename}")

# dirMasked = "E:/CSE/datasetSample/MaskedImage"

import pandas as pd
import matplotlib.pyplot as plt

# Read the Excel file into a pandas DataFrame
data = pd.read_csv('data.csv')

# Get the x-axis values (the row numbers)
x_values = range(1, len(data) + 1)

# Get the y-axis values (the values in the 'Value' column)
y_values = data['l1_loss']

# Create the plot
plt.plot(x_values, y_values)

# Set the labels for the x and y axes
plt.xlabel('Number')
plt.ylabel('Value')

# Set the title of the plot
plt.title('Improvement of Values Over Number')

# Show the plot
plt.show()
