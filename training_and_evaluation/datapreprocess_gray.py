import cv2
import os

def apply_basic_filter(input_folder, output_folder):

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

            # convert the image to grayscale
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # save the filtered image
            cv2.imwrite(output_path, grayscale_image)
            print(f"Filtered image saved: {output_path}")

input_folder = "fiveK/original"
output_folder = "fiveK/grayscale"

apply_basic_filter(input_folder, output_folder)
