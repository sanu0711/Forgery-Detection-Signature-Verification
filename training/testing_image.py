import os
import random
import matplotlib.pyplot as plt
from PIL import Image


# Function to display an image
def display_image(image_path, title):
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')

# Define directories
real_dir = '/content/Train/Real'
fake_dir = '/content/Train/Fake'

# Get the list of original and fake filenames
real_files = [f for f in os.listdir(real_dir) if os.path.isdir(os.path.join(real_dir, f))]
fake_files = [f for f in os.listdir(fake_dir) if f.endswith('_forg') and os.path.isdir(os.path.join(fake_dir, f))]

# Extract base names for matching
real_base_names = set(real_files)
fake_base_names = set(f.split('_forg')[0] for f in fake_files)

# Find common base names
common_names = real_base_names & fake_base_names

if not common_names:
    raise ValueError("No matching original and fake signature pairs found.")

# Randomly select a common name
selected_name = random.choice(list(common_names))

# Get paths for the selected original and fake images
real_image_dir = os.path.join(real_dir, selected_name)
fake_image_dir = os.path.join(fake_dir, selected_name + '_forg')

# Select a random image from each directory
real_image_path = random.choice([os.path.join(real_image_dir, f) for f in os.listdir(real_image_dir) if os.path.isfile(os.path.join(real_image_dir, f))])
fake_image_path = random.choice([os.path.join(fake_image_dir, f) for f in os.listdir(fake_image_dir) if os.path.isfile(os.path.join(fake_image_dir, f))])

# Plot the images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
display_image(real_image_path, 'Original Signature')

plt.subplot(1, 2, 2)
display_image(fake_image_path, 'Fake Signature')

plt.show()