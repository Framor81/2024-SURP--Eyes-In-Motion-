import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


class SCSEBlock(nn.Module):
    def __init__(self, in_channels):
        super(SCSEBlock, self).__init__()
        self.channel_excitation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_excitation = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        chn_se = self.channel_excitation(x)
        spa_se = self.spatial_excitation(x)
        return x * chn_se + x * spa_se


class EyeUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(EyeUNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bridge = conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = conv_block(1024, 512)
        self.scse4 = SCSEBlock(512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block(512, 256)
        self.scse3 = SCSEBlock(256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128)
        self.scse2 = SCSEBlock(128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64)
        self.scse1 = SCSEBlock(64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        self.deep_supervised1 = nn.Conv2d(128, out_channels, kernel_size=1)
        self.deep_supervised2 = nn.Conv2d(256, out_channels, kernel_size=1)
        self.deep_supervised3 = nn.Conv2d(512, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        bridge = self.bridge(self.pool(enc4))

        dec4 = self.upconv4(bridge)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec4 = self.scse4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec3 = self.scse3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec2 = self.scse2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        dec1 = self.scse1(dec1)

        final_output = self.final_conv(dec1)

        deep_supervised_output1 = self.deep_supervised1(dec2)
        deep_supervised_output2 = self.deep_supervised2(dec3)
        deep_supervised_output3 = self.deep_supervised3(dec4)

        return final_output, deep_supervised_output1, deep_supervised_output2, deep_supervised_output3


# Dummy dataset for illustration purposes
class DummyEyeDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform

    def __len__(self):
        return 100  # number of samples

    def __getitem__(self, idx):
        # Dummy data
        image = Image.fromarray(np.random.randint(0, 255, (128, 96, 3), dtype=np.uint8))
        mask = Image.fromarray(np.random.randint(0, 2, (128, 96), dtype=np.uint8))

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


# Training settings
batch_size = 2
learning_rate = 0.001
num_epochs = 5

# Data transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load data
train_dataset = DummyEyeDataset(transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, and optimizer
model = EyeUNet()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        optimizer.zero_grad()
        outputs, _, _, _ = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

# Save the trained model
torch.save(model.state_dict(), 'eye_unet_model.pth')
