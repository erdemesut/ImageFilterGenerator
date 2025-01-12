import cv2
import os
import numpy as np

def apply_artistic_filter(input_folder, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # read the image
            image = cv2.imread(input_path)
            if image is None:
                print(f"Error reading image: {input_path}")
                continue

            # apply the oil painting effect
            oil_painting = cv2.xphoto.oilPainting(image, size=7, dynRatio=1)

            # save the filtered image
            cv2.imwrite(output_path, oil_painting)
            print(f"Filtered image saved: {output_path}")

input_folder = "fiveK/original"
output_folder = "fiveK/oilpainting"

apply_artistic_filter(input_folder, output_folder)
