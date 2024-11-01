# %% Import Libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
import nibabel as nib
import numpy as np
from torchvision import transforms
import time
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from tqdm import tqdm  # For progress bar


# %% Define the Dataset Class for MRI and Lesion Data
class GetDataset(Dataset):
    def __init__(self, MRIFolder, lesionFolder, mri_transform=None, lesion_transform=None):
        self.MRIFolder = MRIFolder
        self.lesionFolder = lesionFolder
        self.mri_transform = mri_transform
        self.lesion_transform = lesion_transform

        self.mri_files = sorted(os.listdir(MRIFolder))
        self.lesion_files = sorted(os.listdir(lesionFolder))

    def __getitem__(self, index):
        mri_path = os.path.join(self.MRIFolder, self.mri_files[index])
        lesion_path = os.path.join(self.lesionFolder, self.lesion_files[index])

        # Load the MRI and lesion NIfTI files
        mri_file = nib.load(mri_path)
        mri_data = mri_file.get_fdata()

        lesion_file = nib.load(lesion_path)
        lesion_data = lesion_file.get_fdata()

        # Apply transformation to MRI data
        if self.mri_transform:
            mri_data = self.mri_transform(mri_data)

        # Apply transformation to lesion mask
        if self.lesion_transform:
            lesion_data = self.lesion_transform(lesion_data)
        else:
            lesion_data = torch.tensor(np.expand_dims(lesion_data, axis=0), dtype=torch.float32)

        return mri_data, lesion_data

    def __len__(self):
        return len(self.mri_files)


# %% Define data transformations
train_mri_transforms = transforms.Compose([
    transforms.Lambda(lambda x: np.expand_dims(x, axis=0)),  # Add channel dimension for MRI data
    transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),  # Convert to PyTorch tensor
    transforms.Normalize(mean=[0], std=[1])  # Normalize intensity of MRI images
])

val_mri_transforms = transforms.Compose([
    transforms.Lambda(lambda x: np.expand_dims(x, axis=0)),
    transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
    transforms.Normalize(mean=[0], std=[1])
])

# %% Prepare Dataset
DATASET_mri = 'MRI'  # Path to MRI Files
DATASET_lesion = 'lesion'  # Path to Lesion Files

# Load the dataset with transformations
dataset = GetDataset(MRIFolder=DATASET_mri, lesionFolder=DATASET_lesion,
                     mri_transform=train_mri_transforms)

# Split the dataset into training and validation sets
train_indices, val_indices = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

# Define DataLoader
batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Define the Attention Block
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # Apply convolutional layers
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Ensure that the spatial dimensions match by interpolating
        if g1.size() != x1.size():
            x1 = F.interpolate(x1, size=g1.shape[2:], mode='trilinear', align_corners=True)

        # Perform the addition and attention
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        # Ensure psi matches the size of x for multiplication
        if psi.size() != x.size():
            psi = F.interpolate(psi, size=x.shape[2:], mode='trilinear', align_corners=True)

        # Apply attention
        return x * psi


# Define the 3D U-Net with Attention Mechanism
class UNet3DWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3DWithAttention, self).__init__()

        def conv_block(in_c, out_c):
            block = nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
            )
            return block

        def up_conv(in_c, out_c):
            return nn.ConvTranspose3d(in_c, out_c, kernel_size=2, stride=2)

        self.encoder1 = conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder3 = conv_block(64, 128)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder4 = conv_block(128, 256)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = conv_block(256, 512)

        self.upconv4 = up_conv(512, 256)
        self.att4 = AttentionBlock(F_g=256, F_l=256, F_int=128)  # Attention block for decoder4
        self.decoder4 = conv_block(512, 256)

        self.upconv3 = up_conv(256, 128)
        self.att3 = AttentionBlock(F_g=128, F_l=128, F_int=64)  # Attention block for decoder3
        self.decoder3 = conv_block(256, 128)

        self.upconv2 = up_conv(128, 64)
        self.att2 = AttentionBlock(F_g=64, F_l=64, F_int=32)  # Attention block for decoder2
        self.decoder2 = conv_block(128, 64)

        self.upconv1 = up_conv(64, 32)
        self.att1 = AttentionBlock(F_g=32, F_l=32, F_int=16)  # Attention block for decoder1
        self.decoder1 = conv_block(64, 32)

        self.final_conv = nn.Conv3d(32, out_channels, kernel_size=1)

    def crop_and_concat(self, encoder_layer, decoder_layer):
        """Helper function to crop encoder layers and concatenate them with decoder layers."""
        enc_size = encoder_layer.size()[2:]
        dec_size = decoder_layer.size()[2:]

        # Calculate cropping based on the difference in size
        crop = [(enc_size[i] - dec_size[i]) // 2 for i in range(len(enc_size))]

        # Crop the encoder layer to match the decoder layer size
        encoder_layer = encoder_layer[:, :, crop[0]:crop[0] + dec_size[0],
                        crop[1]:crop[1] + dec_size[1],
                        crop[2]:crop[2] + dec_size[2]]

        return torch.cat((encoder_layer, decoder_layer), dim=1)

    def forward(self, x):
        # Encoding path
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)

        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)

        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)

        e4 = self.encoder4(p3)
        p4 = self.pool4(e4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoding path with attention gates
        d4 = self.upconv4(b)
        e4 = self.att4(g=d4, x=e4)  # Apply attention gate to skip connection
        d4 = self.crop_and_concat(e4, d4)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        e3 = self.att3(g=d3, x=e3)  # Apply attention gate to skip connection
        d3 = self.crop_and_concat(e3, d3)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        e2 = self.att2(g=d2, x=e2)  # Apply attention gate to skip connection
        d2 = self.crop_and_concat(e2, d2)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        e1 = self.att1(g=d1, x=e1)  # Apply attention gate to skip connection
        d1 = self.crop_and_concat(e1, d1)
        d1 = self.decoder1(d1)

        # Apply padding to match the original input size
        output = self.final_conv(d1)

        # Calculate the difference in size between input and output
        diffZ = x.size()[2] - output.size()[2]
        diffY = x.size()[3] - output.size()[3]
        diffX = x.size()[4] - output.size()[4]

        # Pad the output to match the input size
        output = F.pad(output, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2,
                                diffZ // 2, diffZ - diffZ // 2])

        return output


# %% Dice Loss Definition
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)  # Apply sigmoid to logits
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice


# %% Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model
# 1 input channel for grayscale MRI, 1 output for binary lesion mask segmentation
model = UNet3DWithAttention(in_channels=1, out_channels=1)
model.to(device)

# Loss function and optimizer
bce_loss = nn.BCEWithLogitsLoss()  # Binary segmentation loss
dice_loss = DiceLoss()  # Dice Loss
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Optional learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

# Set up TensorBoard for logging
writer = SummaryWriter(log_dir='runs/brain_lesion_detection')

# Define path for saving checkpoints
checkpoint_dir = 'unet_checkpoints/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


# %% Function to Load Checkpoint
def load_checkpoint(checkpoint_path, model, optimizer=None, strict=True):
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)

    # If the checkpoint contains full model and optimizer states
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
    else:
        # If only model weights are saved
        model.load_state_dict(checkpoint, strict=strict)
        start_epoch = 7
    return model, optimizer, start_epoch


# %% Check if Resuming from Checkpoint
resume_checkpoint = 'unet_checkpoints/3d_unet_epoch_30.pth'  # Change to your checkpoint file if needed
start_epoch = 30
if resume_checkpoint and os.path.exists(resume_checkpoint):
    model, optimizer, start_epoch = load_checkpoint(resume_checkpoint, model, optimizer)
    print(f"Resuming training from epoch {start_epoch + 1}")

# %% Training Loop with Progress Bar and Checkpoints
num_epochs = 100  # You can adjust the number of epochs
train_loss_list = []
val_loss_list = []
torch.cuda.empty_cache()

for epoch in range(start_epoch, num_epochs):
    start_time = time.time()

    # Training phase
    model.train()
    train_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (Training)")  # Progress bar for training
    for batch in progress_bar:
        inputs, labels = batch

        # Move data to GPU if available
        inputs = inputs.to(device).float()
        labels = labels.to(device).float()

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss (combined BCE + Dice loss)
        loss = bce_loss(outputs, labels) + dice_loss(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss
        train_loss += loss.item() * inputs.size(0)
        progress_bar.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(train_loader.dataset)
    train_loss_list.append(avg_train_loss)

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        progress_bar = tqdm(val_loader,
                            desc=f"Epoch {epoch + 1}/{num_epochs} (Validation)")  # Progress bar for validation
        for batch in progress_bar:
            inputs, labels = batch

            # Move data to GPU if available
            inputs = inputs.to(device).float()
            labels = labels.to(device).float()

            # Forward pass
            outputs = model(inputs)

            # Compute loss (combined BCE + Dice loss)
            loss = bce_loss(outputs, labels) + dice_loss(outputs, labels)

            # Accumulate loss
            val_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix(loss=loss.item())

    avg_val_loss = val_loss / len(val_loader.dataset)
    val_loss_list.append(avg_val_loss)

    # Log losses to TensorBoard
    writer.add_scalar('Loss/train', avg_train_loss, epoch)
    writer.add_scalar('Loss/val', avg_val_loss, epoch)

    # Adjust learning rate based on validation loss
    scheduler.step(avg_val_loss)

    # Print epoch statistics
    end_time = time.time()
    print(f"Epoch {epoch + 1}/{num_epochs}, "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, "
          f"Time: {end_time - start_time:.2f}s")

    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'3d_unet_epoch_{epoch + 1}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

# %% Close the TensorBoard writer
writer.close()

# %% Save the final model
torch.save(model.state_dict(), os.path.join(checkpoint_dir, '3d_unet_final.pth'))
