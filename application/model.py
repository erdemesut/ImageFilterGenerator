import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # bottleneck
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        # decoder
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)

        self.output_layer = nn.Conv2d(128, 3, kernel_size=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # encoder
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))  # now image is 128x128

        # bottleneck
        x = self.relu(self.conv3(x))  # still 128x128

        # decoder
        x = self.relu(self.deconv1(x))  # upsample back to 256x256
        x = self.output_layer(x)
        return x
