import pandas as pd
import numpy as np
import os
from PIL import Image

# Load CSV (update path if needed)
test_data = pd.read_csv('sign_mnist_test.csv')

# Make output folder
output_dir = 'test_images'
os.makedirs(output_dir, exist_ok=True)

# Split into labels and pixels
labels = test_data['label']
images = test_data.drop('label', axis=1)

# Label mapping (optional, for filename readability)
label_mapping = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P',
    17: 'Q', 18: 'R', 19: 'S', 20: 'T', 21: 'U', 22: 'V', 23: 'W', 24: 'X', 25: 'Y'
}

# How many images to save
N = 20

# Loop and save as .png
for i in range(N):
    image_array = np.array(images.iloc[i]).reshape(28, 28)
    image = Image.fromarray(np.uint8(image_array), mode='L')
    
    label_num = labels.iloc[i]
    label = label_mapping.get(label_num, str(label_num))
    
    image_path = os.path.join(output_dir, f"{label}_{i}.png")
    image.save(image_path)

print(f"✅ Saved {N} images to: {output_dir}/")
