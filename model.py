from torch import nn

class ConvNet(nn.Module):
    """Class of Convolutional Network model (AlexNet style)."""
    def __init__(self, num_classes):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1) # To avoid overfitting
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Dropout2d(p=0.1)
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(800, 64),
            nn.LeakyReLU()
        )
        self.fc2 = nn.Linear(64, num_classes)

    def num_params(self):
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
