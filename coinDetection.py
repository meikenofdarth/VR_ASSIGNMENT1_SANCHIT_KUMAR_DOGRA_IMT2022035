import cv2
import numpy as np

# Load the image
image_path = "/Users/sanchitkumardogra/kaam/clg/SEM 6/vr/practicals/images/coins/coins1.jpeg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Image not found!")
    exit()

# Apply a stronger Gaussian Blur to smooth the image
blurred = cv2.GaussianBlur(image, (15, 15), 2)

# Use Adaptive Thresholding instead of a fixed threshold
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Perform Morphological Closing to fill gaps in contours
kernel = np.ones((5, 5), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Load original image for visualization
output = cv2.imread(image_path)

# Minimum circularity threshold
min_circularity = 0.3
detected_coins = 0

for cnt in contours:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if perimeter == 0:
        continue  # Avoid division by zero

    circularity = (4 * np.pi * area) / (perimeter ** 2)

    # Adjust area threshold based on coin size
    if circularity > min_circularity and 1000 < area < 50000:
        detected_coins += 1
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        
        # Draw detected circle
        cv2.circle(output, center, radius, (0, 255, 0), 3)
        cv2.circle(output, center, 2, (0, 0, 255), 3)  # Red dot at the center

print(f"Total Coins Detected: {detected_coins}")

# Show results
cv2.imshow("Detected Coins", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

