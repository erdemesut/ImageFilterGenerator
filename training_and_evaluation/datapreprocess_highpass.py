import cv2
import os

def apply_high_pass_filter(input_folder, output_folder, kernel_size):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # read the image
            image = cv2.imread(input_path, cv2.IMREAD_COLOR)
            if image is None:
                print(f"Error reading image: {input_path}")
                continue

            # apply Gaussian blur (low-pass filter)
            blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

            # subtract the blurred image from the original to create a high-pass effect
            high_pass_image = cv2.subtract(image, blurred_image)

            # save the filtered image
            cv2.imwrite(output_path, high_pass_image)
            print(f"Filtered image saved: {output_path}")

input_folder = "dataset/training/original"
output_folder = "dataset/training/filtered"
kernel_size = 15

apply_high_pass_filter(input_folder, output_folder, kernel_size)
