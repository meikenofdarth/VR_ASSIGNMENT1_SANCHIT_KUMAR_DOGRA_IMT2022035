import cv2
import numpy as np

# Define the image paths for the panorama
# Make sure these images have some overlapping regions for better stitching
image_paths = [
    "/Users/sanchitkumardogra/kaam/clg/SEM 6/vr/VR_ASSIGNMENT1/images/panorama/panorama1.jpeg",
    "/Users/sanchitkumardogra/kaam/clg/SEM 6/vr/VR_ASSIGNMENT1/images/panorama/panorama2.jpeg",
    "/Users/sanchitkumardogra/kaam/clg/SEM 6/vr/VR_ASSIGNMENT1/images/panorama/panorama3.jpeg"
]

def load_images(image_paths):
    """
    Loads images from the given file paths and returns them as a list.
    If an image fails to load, it prints an error message.
    """
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
        else:
            print(f"Error loading image: {path}")  # Warn the user if an image is missing or corrupt
    return images

def stitch_images(images):
    """
    Stitches a list of images into a single panorama using OpenCV's built-in stitching function.
    Returns the panorama image if successful, otherwise prints an error.
    """
    stitcher = cv2.Stitcher_create()  # Create a stitcher object to merge images
    (status, panorama) = stitcher.stitch(images)  # Try to stitch images together
    
    if status == cv2.Stitcher_OK:
        print("Panorama stitching successful!")  # Everything went well!
        return panorama
    else:
        print(f"Panorama stitching failed! Error Code: {status}")  # Something went wrong
        return None

def main():
    """
    Main function that loads images, checks if stitching is possible, 
    and saves the final panorama if successful.
    """
    images = load_images(image_paths)

    if len(images) < 2:
        print("At least two images are required for panorama stitching.")
        return  # Exit if not enough images

    panorama = stitch_images(images)

    if panorama is not None:
        output_path = "/Users/sanchitkumardogra/kaam/clg/SEM 6/vr/VR_ASSIGNMENT1/images/panorama/panorama_output.jpeg"
        cv2.imwrite(output_path, panorama)  # Save the stitched panorama
        print(f"Panorama saved at: {output_path}")

        # Display the final stitched panorama
        cv2.imshow("Panorama", panorama)
        cv2.waitKey(0)
        cv2.destroyAllWindows()  # Close the image window when done

# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()
