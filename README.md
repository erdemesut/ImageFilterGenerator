
# **Image Filter Generator**

This project provides an AI-based approach to learn and replicate artistic image filters using Convolutional Neural Networks (CNNs). It includes training, evaluation, and a user-friendly interface to apply filters.

---

## **Setup Instructions**

### **1. Install Dependencies**
Ensure you have Python installed, then install the required libraries:
```bash
pip install -r requirements.txt
```

---

## **Workflow**

### **1. Dataset Preparation**
#### **Flickr30k Dataset**
1. Download the [Flickr30k dataset](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) for training.
2. Preprocess the dataset for the **enhanced sepia filter**:
   - Use `datapreprocess_sepia.py` to preprocess images:
     - The script will split the images into `dataset/training/original` and `dataset/training/filtered` directories.
   ```bash
   python datapreprocess_sepia.py
   ```
3. For training other filters, use the corresponding preprocessing scripts:
   - Example: `datapreprocess_gaussian.py`, `datapreprocess_oilpainting.py`, etc.

#### **Other Datasets**
- For training with datasets other than Flickr30k:
  - Resize the images to 256x256 pixels using `resize.py`

---

### **2. Training**
Train the model using any of the preprocessed datasets:
```bash
python train.py
```
- After training, rename the model file to the corresponding filter name. For example:
  ```bash
  mv filter_model.pth grayscale_model.pth
  ```

---

### **3. Using the Application**
Once you’ve trained and saved the model, you can use `ui.py` to interact with the application:
```bash
python ui.py
```
The UI allows users to:
- Select an image.
- Apply filters using the trained models.
- Compare the AI-generated filter with the manually filtered image.
- Display the SSIM score for the filtered images.

---

### **4. Evaluation**
#### **MIT-Adobe FiveK Dataset**
1. Download the [MIT-Adobe FiveK dataset](https://data.csail.mit.edu/graphics/fivek/) for evaluation.
2. Preprocess the dataset to prepare it for evaluation:
   - Use `datapreprocess_sepia.py` or other preprocessing scripts as required.
   ```bash
   python datapreprocess_sepia.py
   ```
3. Run the evaluation script:
   - Modify folder names in `evaluate_fiveK.py` to point to your datasets.
   ```bash
   python evaluate_fiveK.py
   ```
   - The script will calculate SSIM scores and save the results to text files.

---

## **File Descriptions**

- **`ui.py`**: User interface for applying and testing filters.
- **`train.py`**: Training script for CNN models.
- **`evaluate_fiveK.py`**: Evaluation script for the MIT-Adobe FiveK dataset.
- **`datapreprocess_sepia.py`**: Preprocessing script for the enhanced sepia filter.
- **`datapreprocess_lowpass.py`**: Preprocessing script for Gaussian blur filter.
- **`datapreprocess_highpass.py`**: Preprocessing script for high-pass filter.
- **`datapreprocess_emboss.py`**: Preprocessing script for emboss filter.
- **`datapreprocess_artistic.py`**: Preprocessing script for oil painting filter.
- **`datapreprocess_gray.py`**: Preprocessing script for grayscale filter.
- **`resize.py`**: Script to resize images to 256x256 pixels.
- **`requirements.txt`**: List of required libraries.

---

## **Notes**

- Ensure your datasets are organized as follows:
  ```
  dataset/
  ├── training/
  │   ├── original/  # Original images for training
  │   └── filtered/  # Filtered images for training
  ```

- Rename trained models to match the corresponding filter:
  ```
  grayscale_model.pth
  gaussian_model.pth
  sepia_model.pth
  oilpainting_model.pth
  emboss_model.pth
  highpass_model.pth
  ```

- MIT-Adobe FiveK dataset is required for evaluation only, not training.

---

## **Contributing**
Feel free to fork this repository and submit pull requests for new filters or improvements.

---

## **License**

