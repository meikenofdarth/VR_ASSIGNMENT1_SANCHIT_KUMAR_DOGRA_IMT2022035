import cv2

image_path = "/Users/sanchitkumardogra/kaam/clg/SEM 6/vr/practicals/images/coins/coins1.jpeg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Image not found or cannot be loaded!")
else:
    print("Image loaded successfully.")
    cv2.imshow("Loaded Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
