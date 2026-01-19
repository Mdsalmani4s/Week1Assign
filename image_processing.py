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


# Scale the image by 1.5x
scaled_image = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

# Display the scaled image
plt.imshow(cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB))
plt.title("Scaled Image (1.5x)")
plt.axis("off")
plt.show()


# Define focal length variations
focal_lengths = [50, 100, 200]

# Display different focal lengths
plt.figure(figsize=(12, 4))
for i, f in enumerate(focal_lengths):
    f_matrix = np.array([[f, 0, w//2], [0, f, h//2], [0, 0, 1]])  # Camera intrinsic matrix
    warped_image = cv2.warpPerspective(image, f_matrix, (w, h))

    plt.subplot(1, 3, i+1)
    plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Focal Length: {f}")
    plt.axis("off")

plt.show()
