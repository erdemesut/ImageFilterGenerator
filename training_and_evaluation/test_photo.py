import torch
from PIL import Image
from torchvision import transforms
from model import CNN
import os

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the model and weights
model = CNN().to(device)
model.load_state_dict(torch.load("filter_model.pth", map_location=device))
model.eval()

# same transformations used during training
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# load input image
input_image_path = "image.jpg"
input_image = Image.open(input_image_path).convert("RGB")
input_tensor = transform(input_image).unsqueeze(0).to(device)

# run the model on the input image
with torch.no_grad():
    output_tensor = model(input_tensor)

# convert the output tensor back to a PIL image
output_image = transforms.ToPILImage()(output_tensor.squeeze(0).cpu())

# save or display the output image
output_image_path = "filtered_with_ai.jpg"
output_image.save(output_image_path)
print(f"Filtered image saved at {output_image_path}")

# visually inspect without saving
# output_image.show()
