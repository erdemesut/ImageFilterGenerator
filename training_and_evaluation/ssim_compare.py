import cv2
from skimage.metrics import structural_similarity as ssim

def compare_images(image_path1, image_path2):
    # load the two images
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    # convert the images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # ensure both images are the same size
    if gray1.shape != gray2.shape:
        raise ValueError("Images must have the same dimensions.")

    # compute the Structural Similarity Index (SSIM)
    score, diff = ssim(gray1, gray2, full=True)

    # normalize the difference image for visualization (optional)
    diff = (diff * 255).astype("uint8")

    return score, diff

image1_path = "C:\\Users\\medre\Desktop\\uni archive\\4_1\\grad_project\\project_files\\latex\media\\girl_sepia.jpg"
image2_path = "C:\\Users\\medre\\Desktop\\uni archive\\4_1\\grad_project\\project_files\\latex\media\\girl_ai_9.jpg"

try:
    ssim_score, diff_image = compare_images(image1_path, image2_path)
    print(f"SSIM Score: {ssim_score:.4f}")
except ValueError as e:
    print(e)
