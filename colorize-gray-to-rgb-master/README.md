# Colorizing Grayscale Images using Convolutional Autoencoders

## Overview
This project aims to convert grayscale images to color using **Convolutional Autoencoders (CAE)**. The model is trained to learn the mapping from grayscale to RGB images using the **CIFAR-10 dataset**. Traditional image processing techniques like OpenCV's `COLOR_RGB2GRAY` work well for grayscale conversion, but this project demonstrates how deep learning can also achieve similar results.

## Why CIFAR-10?
- CIFAR-10 contains diverse image categories, making it a robust dataset for training.
- The images are small (32x32x3), allowing faster training and experimentation.
- The method used here can be scaled for higher-resolution images.

## Dataset
- Each image is of size **32x32x3**.
- The dataset is downloaded from the **CIFAR-10 website**.
- **Torchvision's built-in dataloaders** are used for easy handling.

## Data Preprocessing
- Convert **RGB images to grayscale** using OpenCV's `COLOR_RGB2GRAY`.
- Split dataset into **80% training and 20% testing**.
- Use **batch size of 100** for efficient training.

## Model Architecture
The **Convolutional Autoencoder** consists of:
### **Encoder:**
- Two convolutional layers with max-pooling.
- Extracts important features from grayscale images.

### **Decoder:**
- Two deconvolutional layers for upsampling.
- Fully connected layer to output **RGB images**.

```python
class ConvNet(nn.Module):
    def __init__(self, batch_size):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.linear1 = nn.Linear(32*7*7, 512)
        self.linear2 = nn.Linear(512, 3072)

    def forward(self, x):
        p1 = F.max_pool2d(self.conv1(x), (2,2))
        p2 = F.max_pool2d(self.conv2(p1), (2,2))
        f1 = F.relu(self.linear1(p2.view(batch_size,-1)))
        f2 = torch.sigmoid(self.linear2(f1)).view(-1, 3, 32, 32)
        return f2
```

## Training Details
- **Batch size:** 100
- **Learning rate:** 0.001
- **Epochs:** 1000
- **Optimizer:** Adam (without weight decay)
- Model **underfits slightly**, which can be improved with deeper networks.

## Results
- The model successfully colorizes grayscale images.
- The generated images are slightly blurry, requiring further enhancements.
- Increasing model complexity could improve output sharpness.

## Future Improvements
- Use **GANs (Generative Adversarial Networks)** for better realism.
- Experiment with **Transformer-based vision models**.
- Train on higher-resolution datasets for more detailed colorization.

## Conclusion
This project demonstrates the effectiveness of **Convolutional Autoencoders** in **grayscale-to-color image translation**. Although the model produces reasonable results, further refinements can enhance the output quality.

---

Feel free to explore and modify the model for better performance! 🚀

