import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('images/elon.jpg')  # Replace with your actual image file
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format

# Display the image
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()



# ===== ROTATION =====
# Define rotation parameters
(h, w) = image.shape[:2]  # Get image height and width
center = (w // 2, h // 2)  # Find the center of the image
angle = 45  # Rotate by 45 degrees
scale = 1.0  # Keep the scale the same

# Create the rotation matrix
rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

# Apply the rotation
rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

# Display the rotated image
plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
plt.title("Rotated Image (45°)")
plt.axis("off")
plt.show()

# ===== SCALING (FIXED) =====
# Scale the image by 1.5x
scale_factor = 1.5
new_width = int(w * scale_factor)
new_height = int(h * scale_factor)
scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

# Display the scaled image (showing it's actually bigger)
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB))
plt.title(f"Scaled Image (1.5x) - New size: {new_width}x{new_height}")
plt.axis("off")
plt.show()

# ===== 3D VISION - FOCAL LENGTH SIMULATION (FIXED) =====
# Simulate different focal lengths (camera zoom effects)
focal_lengths = [50, 100, 200]

plt.figure(figsize=(15, 5))
for i, focal_length in enumerate(focal_lengths):
    # Calculate crop factor based on focal length
    # Lower focal length = wider view (zoom out)
    # Higher focal length = narrower view (zoom in)
    crop_factor = 100 / focal_length
    
    # Calculate crop dimensions
    crop_w = int(w * crop_factor)
    crop_h = int(h * crop_factor)
    
    # Calculate crop coordinates (center crop)
    x_start = (w - crop_w) // 2
    y_start = (h - crop_h) // 2
    
    # Crop the image
    cropped = image[y_start:y_start+crop_h, x_start:x_start+crop_w]
    
    # Resize back to original dimensions for display
    focal_image = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Display
    plt.subplot(1, 3, i+1)
    plt.imshow(cv2.cvtColor(focal_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Focal Length: {focal_length}mm")
    plt.axis("off")

plt.tight_layout()
plt.show()

print("\n✅ All transformations completed successfully!")
print(f"Original image size: {w}x{h}")
print(f"Scaled image size: {new_width}x{new_height}")