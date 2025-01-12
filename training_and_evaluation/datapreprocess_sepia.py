from PIL import Image, ImageEnhance
import os

# input directory containing the original images
input_folder = "fiveK/fiveK_dataset"

# output directories
output_original_folder = "dataset/training/original"
output_filtered_folder = "dataset/training/filtered"

# desired size (width, height)
desired_size = (256, 256)

# create the output directories if they don't exist
os.makedirs(output_original_folder, exist_ok=True)
os.makedirs(output_filtered_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        input_path = os.path.join(input_folder, filename)

        # open the image
        img = Image.open(input_path).convert("RGB")

        # resize the image
        img_resized = img.resize(desired_size, resample=Image.Resampling.LANCZOS)

        # save the resized original image
        original_output_path = os.path.join(output_original_folder, filename)
        img_resized.save(original_output_path)

        # applying a filter
        # increase contrast
        enhancer = ImageEnhance.Contrast(img_resized)
        img_filtered = enhancer.enhance(1.8)

        # increase Saturation
        enhancer_color = ImageEnhance.Color(img_filtered)
        img_filtered = enhancer_color.enhance(1.5)

        # applying a Sepia tone
        # convert to R/G/B and apply a sepia transformation
        r, g, b = img_filtered.split()

        # Convert to sepia by applying a formula:
        r = r.point(lambda i: i * 0.393 + i * 0.769 + i * 0.189)
        g = g.point(lambda i: i * 0.349 + i * 0.686 + i * 0.168)
        b = b.point(lambda i: i * 0.272 + i * 0.534 + i * 0.131)

        # Merge back into an RGB image
        # We'll just shift the tones toward warm colors after re-merging.

        img_sepia_approx = Image.merge("RGB", (r, g, b))

        # save the filtered image
        filtered_output_path = os.path.join(output_filtered_folder, filename)
        img_filtered.save(filtered_output_path)

print("All images resized and filtered.")
