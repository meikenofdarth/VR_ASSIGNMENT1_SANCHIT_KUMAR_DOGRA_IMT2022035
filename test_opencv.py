import cv2

# Load an image
image = cv2.imread("/Users/sanchitkumardogra/kaam/clg/SEM 6/vr/practicals/test.png")  # Replace with your image path

# Show the image
cv2.imshow("Test Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

