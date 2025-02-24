# Only use this repo, to download the images too for the assignment, do not use the zip submitted on lms. Zip size exceeded 20 mb hence had to remove all images from the submission on LMS
# VR Assignment 1: Coin Detection and Panorama Creation
## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [How to Run](#how-to-run)
  - [Coin Detection](#coin-detection)
  - [Panorama Creation](#panorama-creation)
- [Explanation of Code](#explanation-of-code)
  - [Coin Detection](#coin-detection-explanation)
  - [Panorama Creation](#panorama-creation-explanation)
- [Sample Outputs](#sample-outputs)
- [Observations](#observations)
- [Dependencies](#dependencies)
- [Conclusion](#conclusion)

---

## Overview

This project consists of two main tasks:

1. **Coin Detection:** Identifying and segmenting coins from an image using image processing techniques.
2. **Panorama Creation:** Stitching multiple images together to form a seamless panoramic view.

Both tasks are implemented in Python using OpenCV and NumPy.

---

## Requirements

Ensure that the following dependencies are installed:

- Python 3.9
- OpenCV (`cv2`)
- NumPy

---

## Installation

To set up the environment, install the required packages:

```sh
pip install opencv-python numpy imutils
```

If using Conda:

```sh
conda install -c conda-forge opencv numpy imutils
```

---

## How to Run

### Coin Detection

Run the following command:

```sh
python coinDetection.py
```

The script will process the image, detect coins, and display the results.

### Panorama Creation

Run the following command:

```sh
python panoramaFormation.py
```

This will stitch the images together and display the final panorama.

---

## Explanation of Code

### Coin Detection Explanation

1. **Loading the Image**: The image is loaded in grayscale mode for processing.
2. **Gaussian Blur**: A stronger Gaussian blur is applied to reduce noise and enhance object edges.
3. **Adaptive Thresholding**: Dynamically segments the coins from the background.
4. **Morphological Operations**: The closing operation is used to fill small gaps and improve contour detection.
5. **Contour Detection**: Extracts potential coin-like shapes from the image.
6. **Circularity Check**: Ensures only circular objects are considered as coins.
7. **Filtering by Area**: Removes small noise and large unwanted objects based on area thresholds.
8. **Drawing Circles**: Highlights detected coins with green circles and their centers with red dots.
9. **Displaying Results**: Outputs the number of detected coins and saves the final image.

### Panorama Creation Explanation

1. **Loading Images**: Reads the input images.
2. **Stitching Process**: Uses SIFT for detecting keypoints and computing descriptors.
3. **Homography**: Finds the homography matrix using cv2.findHomography().
4. **Warping**: Warps one image onto the other using cv2.warpPerspective()
5. **Handling Errors**: Checks if stitching was successful and handles failures.
6. **Saving and Displaying**: Outputs the final panorama image.

---

## Sample Inputs

### Coin Detection Inputs

This picture of 8 coins was taken from the internet.
![Coin Detection Input](images/coins/coins1.jpeg)

### Panorama Creation Inputs

These 2 pictures of my work environment were taken using my phone's camera.
![Panorama Input](images/panorama/panorama1.jpeg)
![Panorama Input](images/panorama/panorama2.jpeg)

---

## Sample Outputs

### Coin Detection Output

The detected coins are highlighted with green circles, and their centers are marked with red dots. The latest method improves upon previous results, now correctly detecting **7 out of 8 coins**. Further improvements are suggested in the Observations section.
![Coin Detection Output](images/coins/output/detected_coins.png)

### Panorama Creation Output

The final panorama image seamlessly combines multiple input images.
![Panorama Output](images/panorama/panorama_output.jpeg)

---

## Observations

### Coin Detection

- **Performance**: Works well when coins are distinct and well-lit.
- **Challenges**:
  - **Overlapping Coins**: Causes merging of contours, leading to undercounting.
  - **Poor Lighting Conditions**: Shadows or reflections can interfere with detection.
  - **Similar Background**: Coins with similar colors to the background may not be segmented well.
- **Solutions Tried**:
  - **Increased Blur Strength**: Helps smooth edges but can reduce fine details.
  - **Adaptive Thresholding**: Improved detection but still misses some coins.
  - **Morphological Closing**: Helped reduce gaps but introduced some unwanted noise.

### Panorama Creation

- **Performance**: Works well with images having good overlap.
- **Challenges**:
  - **Lighting Differences**: Affects blending quality.
  - **Non-Overlapping Areas**: Causes stitching errors or black patches.
- **Possible Enhancements**:
  - Use **feature-based matching** for more robust stitching.
  - Try **exposure compensation** for better blending.
---

## Conclusion

This project successfully demonstrates fundamental image processing techniques for object detection and panorama stitching using OpenCV. Enhancements like deep learning-based detection or feature matching could further improve accuracy.

