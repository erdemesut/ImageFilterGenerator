import os
import torch
from PIL import Image
from torchvision import transforms
from pytorch_msssim import ssim
from model import CNN
import matplotlib.pyplot as plt


model_path = "sepia_model.pth"  # Path to trained model
input_folder = "fiveK/original"  # Folder containing original images
output_folder = "fiveK/enhanced_sepia_ai"  # Folder to save AI-filtered images
manual_filtered_folder = "fiveK/enhanced_sepia"  # Folder with manually filtered images

os.makedirs(output_folder, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the model
model = CNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


transform = transforms.Compose([
    #transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
reverse_transform = transforms.ToPILImage()

# process images
ssim_scores = []
for image_name in os.listdir(input_folder):
    input_image_path = os.path.join(input_folder, image_name)
    manual_filtered_path = os.path.join(manual_filtered_folder, image_name)

    if not os.path.exists(manual_filtered_path):
        print(f"Manual filtered image not found for: {image_name}")
        continue

    try:
        # open and preprocess the input image
        input_image = Image.open(input_image_path).convert("RGB")
        input_tensor = transform(input_image).unsqueeze(0).to(device)

        # apply the model
        with torch.no_grad():
            output_tensor = model(input_tensor)

        # convert output tensor to image
        output_image = reverse_transform(output_tensor.squeeze(0).cpu())

        # save the ai-filtered image
        output_image_path = os.path.join(output_folder, image_name)
        output_image.save(output_image_path)

        # open the manually filtered image
        manual_filtered_image = Image.open(manual_filtered_path).convert("RGB")
        manual_filtered_tensor = transform(manual_filtered_image).unsqueeze(0).to(device)

        # calculate SSIM score
        score = ssim(output_tensor, manual_filtered_tensor, data_range=1).item()
        ssim_scores.append((image_name, score))
        print(f"Processed {image_name}: SSIM = {score:.4f}")

    except Exception as e:
        print(f"Error processing {image_name}: {e}")

# save SSIM scores to a file
ssim_output_path = os.path.join(output_folder, "ssim_scores.txt")
with open(ssim_output_path, "w") as f:
    for image_name, score in ssim_scores:
        f.write(f"{image_name}: {score:.4f}\n")

# calculate average SSIM and percentage above 80%
ssim_values = [score for _, score in ssim_scores]
average_ssim = sum(ssim_values) / len(ssim_values)
percentage_above_80 = sum(1 for s in ssim_values if s > 0.8) / len(ssim_values) * 100

print(f"Average SSIM: {average_ssim:.4f}")
print(f"Percentage of images with SSIM > 80%: {percentage_above_80:.2f}%")

# save summary
summary_output_path = os.path.join(output_folder, "ssim_summary.txt")
with open(summary_output_path, "w") as f:
    f.write(f"Average SSIM: {average_ssim:.4f}\n")
    f.write(f"Percentage of images with SSIM > 80%: {percentage_above_80:.2f}%\n")

# plot SSIM scores
plt.figure(figsize=(10, 6))
plt.plot(range(len(ssim_values)), ssim_values, marker='o', linestyle='-', markersize=2)
plt.title("SSIM Scores for All Images")
plt.xlabel("Image Index")
plt.ylabel("SSIM Score")
plt.grid(True)
plt.savefig(os.path.join(output_folder, "ssim_scores_plot.png"))
plt.show()

print(f"Processing complete. SSIM scores and summary saved to {output_folder}.")
