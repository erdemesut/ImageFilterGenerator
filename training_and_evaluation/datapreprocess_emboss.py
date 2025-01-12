import cv2
import os
import numpy as np
def apply_emboss_filter(input_folder, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # read the image
            image = cv2.imread(input_path, cv2.IMREAD_COLOR)
            if image is None:
                print(f"Error reading image: {input_path}")
                continue

            # convert to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # apply the emboss kernel
            emboss_kernel = np.array([[-2, -1, 0],
                                       [-1,  1, 1],
                                       [ 0,  1, 2]])

            embossed_image = cv2.filter2D(gray_image, -1, emboss_kernel)

            # normalize the result to 0-255
            embossed_image = cv2.normalize(embossed_image, None, 0, 255, cv2.NORM_MINMAX)

            # save the embossed image
            cv2.imwrite(output_path, embossed_image)
            print(f"Filtered image saved: {output_path}")

input_folder = "dataset/training/original"
output_folder = "dataset/training/filtered"

apply_emboss_filter(input_folder, output_folder)
