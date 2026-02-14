import torch
import torch.nn as nn

class CustomOPN(nn.Module):
    def __init__(self):
        super(CustomOPN, self).__init__()

        # 1. FEATURE EXTRACTION
        # Processes the input frames and extracts features, before flattening output to next stage.
        self.conv_layers = nn.Sequential(
        nn.Conv2d(4*3, 96, kernel_size=11, stride=4, padding=0),  # accepting frames*channels as input
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75),
        nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75),
        nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(384),
        nn.ReLU(inplace=True),
        nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=2),
        nn.BatchNorm2d(384),
        nn.ReLU(inplace=True),
        nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=2),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # 2. PAIRWISE FEATURE EXTRACTION:
        # fc6 layer processes flattened output, splits it into four parts, 
        # and concatenates pairs to form new feature vectors. These vectors are then processed by the fc7 layers.
        self.fc6 = nn.Linear(256 * 3 * 3, 1024)  # Assuming the input size from conv2d is (256, 3, 3)
        self.bn6 = nn.BatchNorm1d(1024)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout(0.5)

        # The fc7 layers process the pairwise features
        self.fc7_layers = nn.ModuleList()
        for i in range(6):
            self.fc7_layers.append(nn.Sequential(
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5)
            ))

        # 3. ORDER PREDICTION:
        # fc8 layer processes concatenated feature vectors from previous stage and outputs final predictions.
        self.fc8 = nn.Linear(512*6, 12)  # 6 is the number of concatenated layers

    def forward(self, x):
        # Feature extraction
        x = self.conv_layers(x)
        #print(f'After conv_layers: {x.shape}')
        x = x.view(x.size(0), -1)  # Flatten the tensor
        #print(f'After flattening: {x.shape}')

        # Pairwise feature extraction
        x = self.fc6(x)
        #print(f'After fc6: {x.shape}')
        x = self.bn6(x)
        x = self.relu6(x)
        x = self.drop6(x)
        x1, x2, x3, x4 = x.chunk(4, dim=1)  # Slice the tensor into 4 parts
        x_concat = [torch.cat((x1, x2), dim=1), torch.cat((x2, x3), dim=1), torch.cat((x3, x4), dim=1),
                torch.cat((x1, x3), dim=1), torch.cat((x2, x4), dim=1), torch.cat((x1, x4), dim=1)]
        
        out = []
        for i in range(6):
            out_i = self.fc7_layers[i](x_concat[i])
            out.append(out_i)

        # Order prediction. Predicts the order of the frames.
        out = torch.cat(out, dim=1)  # Concatenate along the channel dimension
        #print(out.shape)
        out = self.fc8(out)
        return out