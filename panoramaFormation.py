import cv2
import numpy as np

# Define the image paths
image_paths = [
    "/Users/sanchitkumardogra/kaam/clg/SEM 6/vr/practicals/images/panorama/panorama1.jpeg",
    "/Users/sanchitkumardogra/kaam/clg/SEM 6/vr/practicals/images/panorama/panorama2.jpeg",
    "/Users/sanchitkumardogra/kaam/clg/SEM 6/vr/practicals/images/panorama/panorama3.jpeg"
]

def load_images(image_paths):
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
        else:
            print(f"Error loading image: {path}")
    return images

def stitch_images(images):
    stitcher = cv2.Stitcher_create()
    (status, panorama) = stitcher.stitch(images)
    
    if status == cv2.Stitcher_OK:
        print("Panorama stitching successful!")
        return panorama
    else:
        print(f"Panorama stitching failed! Error Code: {status}")
        return None

def main():
    images = load_images(image_paths)
    
    if len(images) < 2:
        print("At least two images are required for panorama stitching.")
        return

    panorama = stitch_images(images)
    
    if panorama is not None:
        output_path = "/Users/sanchitkumardogra/kaam/clg/SEM 6/vr/practicals/images/panorama/panorama_output.jpeg"
        cv2.imwrite(output_path, panorama)
        print(f"Panorama saved at: {output_path}")

        cv2.imshow("Panorama", panorama)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
