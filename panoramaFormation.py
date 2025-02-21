import numpy as np
import imutils
import cv2
import os

# Function to save an image to a specified path
def save_image(image, path):
    cv2.imwrite(path, image)

# Function to detect keypoints and compute descriptors using SIFT
def sift_detect_descriptor(image, save_path):
    descriptor = cv2.SIFT_create()
    kps, features = descriptor.detectAndCompute(image, None)
    
    # Draw keypoints on the image for visualization
    img_with_kps = cv2.drawKeypoints(image, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    save_image(img_with_kps, save_path)
    
    return np.float32([kp.pt for kp in kps]), features

# Function to match keypoints between two images using BFMatcher and ratio test
def interest_point_matcher(imageA, imageB, interestA, interestB, xA, xB, ratio, re_proj, save_path):
    matcher = cv2.BFMatcher()
    rawMatches = matcher.knnMatch(xA, xB, 2)
    matches = []
    
    # Apply Lowe's ratio test to filter good matches
    for m in rawMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append(m[0])

    if len(matches) > 4:
        ptsA = np.float32([interestA[m.queryIdx] for m in matches])
        ptsB = np.float32([interestB[m.trainIdx] for m in matches])
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, re_proj)

        # Convert interest points to KeyPoint objects for visualization
        keypointsA = [cv2.KeyPoint(f[0], f[1], 1) for f in interestA]
        keypointsB = [cv2.KeyPoint(f[0], f[1], 1) for f in interestB]


        # Draw matches correctly
        img_matches = cv2.drawMatches(imageA, keypointsA, imageB, keypointsB, matches, None)
        save_image(img_matches, save_path)

        return matches, H, status
    
    return None


# Function to crop extra black regions from the stitched panorama
def crop_black_region(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        return image[y:y+h-1, x:x+w-1]
    
    return image

# Function to perform image stitching
def stitch(images, ratio=0.75, re_proj=5.0):
    imageB, imageA = images
    
    # Save the original input images
    save_image(imageA, "images/panorama/output/original1.jpg")
    save_image(imageB, "images/panorama/output/original2.jpg")
    
    # Detect keypoints and descriptors
    interestA, xA = sift_detect_descriptor(imageA, "images/panorama/output/sift1.jpg")
    interestB, xB = sift_detect_descriptor(imageB, "images/panorama/output/sift2.jpg")
    
    # Match keypoints and find the homography matrix
    M = interest_point_matcher(imageA, imageB, interestA, interestB, xA, xB, ratio, re_proj, "images/panorama/output/matches.jpg")
    if M is None:
        print("Not enough matches found.")
        return None
    
    matches, H, status = M
    
    # Warp one image onto the other using the homography matrix
    pano_img = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
    save_image(pano_img, "images/panorama/output/warped.jpg")
    
    # Overlay the second image onto the panorama
    pano_img[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
    
    # Crop unnecessary black regions
    pano_img = crop_black_region(pano_img)
    save_image(pano_img, "images/panorama/output/panorama.jpg")
    
    return pano_img

# Define image file paths
image_paths = [
    "/Users/sanchitkumardogra/kaam/clg/SEM 6/vr/VR_ASSIGNMENT1/images/panorama/panorama1.jpeg",
    "/Users/sanchitkumardogra/kaam/clg/SEM 6/vr/VR_ASSIGNMENT1/images/panorama/panorama2.jpeg"
]

# Load images from disk
image1 = cv2.imread(image_paths[0])
image2 = cv2.imread(image_paths[1])

if image1 is None or image2 is None:
    print("Error loading images. Check the paths.")
else:
    # Resize images for faster processing
    image1 = imutils.resize(image1, width=600)
    image2 = imutils.resize(image2, width=600)
    
    # Perform image stitching
    stitched_image = stitch([image1, image2])
    
    if stitched_image is not None:
        output_path = "/Users/sanchitkumardogra/kaam/clg/SEM 6/vr/VR_ASSIGNMENT1/images/panorama/output/stitched_panorama.jpg"
        cv2.imwrite(output_path, stitched_image)
        print(f"Stitched image saved at: {output_path}")
    else:
        print("Stitching failed.")
