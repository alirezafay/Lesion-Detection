import torch
import torch.nn as nn

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

# Define a function to load the model
def load_model(model_path, model):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))  # Load the checkpoint file
    model.load_state_dict(checkpoint['model_state_dict'])  # Load the state dictionary into the model
    print(f"Model loaded successfully from '{model_path}'")
    return model

# Create an instance of your model and load the weights
model = UNet3DWithAttention(in_channels=1, out_channels=1)
model_path = 'model.pth'

# Load the model
model = load_model(model_path, model)
