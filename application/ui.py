import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageEnhance
import cv2
import numpy as np
import torch
from torchvision import transforms
from model import CNN
from pytorch_msssim import ssim

# initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# trained models
models = {
    "Grayscale": "grayscale_model.pth",
    "Gaussian Blur": "gaussian_model.pth",
    "Highpass": "highpass_model.pth",
    "Enhanced Sepia": "sepia_model.pth",
    "Oil Painting": "oilpainting_model.pth",
    "Emboss": "emboss_model.pth"
}

# load models into memory
loaded_models = {}
for filter_name, model_path in models.items():
    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    loaded_models[filter_name] = model

# image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# reverse transform for displaying filtered image
reverse_transform = transforms.ToPILImage()

# main application
class FilterApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Filter Application")

        self.original_image = None
        self.manual_filtered_image = None
        self.filtered_image = None

        self.setup_ui()

    def setup_ui(self):
        # select image button
        self.select_button = tk.Button(self.root, text="Select Image", command=self.select_image)
        self.select_button.pack(pady=10)

        # image display frames
        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack()

        self.original_image_label = tk.Label(self.image_frame, text="")
        self.original_image_label.grid(row=0, column=0, padx=10, pady=10)
        self.original_image_caption = tk.Label(self.image_frame, text="Original Image")
        self.original_image_caption.grid(row=1, column=0)

        self.manual_filtered_image_label = tk.Label(self.image_frame, text="")
        self.manual_filtered_image_label.grid(row=0, column=1, padx=10, pady=10)
        self.manual_filtered_caption = tk.Label(self.image_frame, text="Manually Filtered")
        self.manual_filtered_caption.grid(row=1, column=1)

        self.filtered_image_label = tk.Label(self.image_frame, text="")
        self.filtered_image_label.grid(row=0, column=2, padx=10, pady=10)
        self.filtered_image_caption = tk.Label(self.image_frame, text="Filtered with Model")
        self.filtered_image_caption.grid(row=1, column=2)

        # filter selection
        self.filter_var = tk.StringVar(value="Grayscale")
        self.filter_menu = tk.OptionMenu(self.root, self.filter_var, *models.keys())
        self.filter_menu.pack(pady=10)

        # apply filter button
        self.apply_button = tk.Button(self.root, text="Apply Filter", command=self.apply_filter)
        self.apply_button.pack(pady=10)

        # SSIM
        self.ssim_label = tk.Label(self.root, text="SSIM Score: N/A")
        self.ssim_label.pack(pady=10)

    def select_image(self):
        # open file dialog to select image
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if not file_path:
            return

        try:
            # open and display the selected image
            self.original_image = Image.open(file_path).convert("RGB")
            resized_image = self.original_image.resize((256, 256))
            tk_image = ImageTk.PhotoImage(resized_image)
            self.original_image_label.config(image=tk_image, text="")
            self.original_image_label.image = tk_image
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image: {e}")

    def apply_filter(self):
        if not self.original_image:
            messagebox.showerror("Error", "Please select an image first!")
            return

        # get the selected filter
        filter_name = self.filter_var.get()
        model = loaded_models.get(filter_name)

        if not model:
            messagebox.showerror("Error", f"Filter '{filter_name}' not found!")
            return

        try:
            # preprocess the image
            input_tensor = transform(self.original_image).unsqueeze(0).to(device)

            # manually apply the selected filter
            self.manual_filtered_image = self.apply_manual_filter(filter_name, self.original_image)

            # display manually filtered image
            manual_tk_image = ImageTk.PhotoImage(self.manual_filtered_image)
            self.manual_filtered_image_label.config(image=manual_tk_image, text="")
            self.manual_filtered_image_label.image = manual_tk_image

            # apply the model
            with torch.no_grad():
                output_tensor = model(input_tensor)

            # convert output tensor to image
            model_filtered_image = reverse_transform(output_tensor.squeeze(0).cpu())

            # display the model filtered image
            self.filtered_image = model_filtered_image
            tk_image = ImageTk.PhotoImage(model_filtered_image)
            self.filtered_image_label.config(image=tk_image, text="")
            self.filtered_image_label.image = tk_image

            # calculate SSIM score between manually filtered and ai-filtered images
            manual_tensor = transform(self.manual_filtered_image).unsqueeze(0).to(device)
            ssim_score = ssim(output_tensor, manual_tensor, data_range=1).item()
            self.ssim_label.config(text=f"SSIM Score: {ssim_score:.4f}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply filter: {e}")

    def apply_manual_filter(self, filter_name, image):
        image_cv = np.array(image)
        if filter_name == "Grayscale":
            return Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)).convert("RGB")
        elif filter_name == "Gaussian Blur":
            return Image.fromarray(cv2.GaussianBlur(image_cv, (15, 15), 0))
        elif filter_name == "Highpass":
            blurred = cv2.GaussianBlur(image_cv, (15, 15), 0)
            highpass = cv2.subtract(image_cv, blurred)
            return Image.fromarray(highpass)
        elif filter_name == "Enhanced Sepia":
            enhancer = ImageEnhance.Contrast(image)
            img_contrast = enhancer.enhance(1.8)
            enhancer_color = ImageEnhance.Color(img_contrast)
            img_saturated = enhancer_color.enhance(1.5)
            r, g, b = img_saturated.split()
            r = r.point(lambda i: i * 0.393 + i * 0.769 + i * 0.189)
            g = g.point(lambda i: i * 0.349 + i * 0.686 + i * 0.168)
            b = b.point(lambda i: i * 0.272 + i * 0.534 + i * 0.131)
            return Image.merge("RGB", (r, g, b))
        elif filter_name == "Oil Painting":
            return Image.fromarray(cv2.xphoto.oilPainting(image_cv, 7, 1))
        elif filter_name == "Emboss":
            gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
            kernel = np.array([[-2, -1, 0],
                               [-1,  1, 1],
                               [ 0,  1, 2]])
            embossed = cv2.filter2D(gray, -1, kernel)
            embossed_normalized = cv2.normalize(embossed, None, 0, 255, cv2.NORM_MINMAX)
            return Image.fromarray(embossed_normalized).convert("RGB")
        else:
            raise ValueError(f"Unknown filter: {filter_name}")

# run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = FilterApplication(root)
    root.mainloop()
