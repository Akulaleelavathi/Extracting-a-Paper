import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = './img2.jpg'  
image = cv2.imread(image_path)
org_image = image.copy()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise and improve edge detection
blurred = cv2.GaussianBlur(gray, (9, 9), 0)

# Detect edges using Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Dilate the edges to increase the search area
kernel = np.ones((5, 5), np.uint8)
dilated_edges = cv2.dilate(edges, kernel, iterations=2)

# Find contours in the dilated edge-detected image
contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours based on area and keep the largest one, assuming it's the paper border
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# If no contours are found, raise an error
if len(contours) == 0:
    raise ValueError("Could not find any contours in the image.")

# Use the largest contour
largest_contour = contours[0]

# Approximate the contour to get the corners (This step tries to detect four corners)
epsilon = 0.02 * cv2.arcLength(largest_contour, True)
approx = cv2.approxPolyDP(largest_contour, epsilon, True)




# A helper function is defined to order the points (top-left, top-right, bottom-right, bottom-left).
# Function to order points
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# Draw contours on the original image
# The largest contour is drawn on a copy of the original image to visualize it.
contour_image = image.copy()
cv2.drawContours(contour_image, [largest_contour], -1, (0, 255, 0), 2)

# Check if we have four points (i.e., four corners)
if len(approx) == 4:
    # Extract the four points
    ordered_points = order_points(approx.reshape(4, 2))
else:
    # If not, use the minimum enclosing rectangle
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    ordered_points = order_points(box)

# The dimensions of the new image (width and height) are computed based on the distances between the corners.
# Compute the width and height of the new image based on the corners
widthA = np.sqrt(((ordered_points[2][0] - ordered_points[3][0]) ** 2) + ((ordered_points[2][1] - ordered_points[3][1]) ** 2))
widthB = np.sqrt(((ordered_points[1][0] - ordered_points[0][0]) ** 2) + ((ordered_points[1][1] - ordered_points[0][1]) ** 2))
maxWidth = max(int(widthA), int(widthB))

heightA = np.sqrt(((ordered_points[1][0] - ordered_points[2][0]) ** 2) + ((ordered_points[1][1] - ordered_points[2][1]) ** 2))
heightB = np.sqrt(((ordered_points[0][0] - ordered_points[3][0]) ** 2) + ((ordered_points[0][1] - ordered_points[3][1]) ** 2))
maxHeight = max(int(heightA), int(heightB))

# Set destination points for the perspective transform
dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype="float32")

# Compute the perspective transform matrix and then apply it
M = cv2.getPerspectiveTransform(ordered_points, dst)
warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

# Resize the warped image to match the original image dimensions
warped_resized = cv2.resize(warped, (image.shape[1], image.shape[0]))

# Save the final warped and resized image
cv2.imwrite('./warped_image.jpg', warped_resized)

# Display the original image with contours and the final warped and resized image
plt.figure(figsize=(12, 6))

# Display the image with contours
plt.subplot(1, 2, 1)
plt.title("Image with Contours")
plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))

# Display the warped and resized image
plt.subplot(1, 2, 2)
plt.title("Warped and Resized Image")
plt.imshow(cv2.cvtColor(warped_resized, cv2.COLOR_BGR2RGB))

plt.show()


