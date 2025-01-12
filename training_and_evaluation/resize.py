from PIL import Image
import os


def resize_images_in_folder(input_folder, output_folder, size=(256, 256)):
    # ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # skip non-image files
        try:
            with Image.open(input_path) as img:
                # resize the image
                resized_img = img.resize(size)

                # save the resized image to the output folder
                output_path = os.path.join(output_folder, filename)
                resized_img.save(output_path)

                print(f"Resized and saved: {filename}")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")


input_folder = "dataset/data"
output_folder = "dataset/training/original"

resize_images_in_folder(input_folder, output_folder)
