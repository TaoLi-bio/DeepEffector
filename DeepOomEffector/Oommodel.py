import torch.nn as nn
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 18
torch.manual_seed(seed)
np.random.seed(seed)


class Model(nn.Module):
    def __init__(self, input_dim1=1280, hidden_dim=640, num_classes=1):
        super(Model, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(input_dim1, num_heads=5)
        self.dropout1 = nn.Dropout(0.1)


        self.residual_conv1 = nn.Conv1d(input_dim1, hidden_dim, kernel_size=1)
        self.conv11 = nn.Conv1d(input_dim1, hidden_dim, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv12 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

        self.conv13 = nn.Conv1d(hidden_dim, hidden_dim//2, kernel_size=3, padding=1)
        self.conv14 = nn.Conv1d(hidden_dim//2, hidden_dim//2, kernel_size=3, padding=1)
        self.residual_conv2 = nn.Conv1d(hidden_dim, hidden_dim//2, kernel_size=3, padding=1)

        self.fc_final2 = nn.Linear(hidden_dim//2, num_classes)

        self.mlp = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim//2, hidden_dim // 2),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim // 2, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attened, _ = self.multihead_attention(x, x, x)
        x1 = self.dropout1(attened)
        x11 = x1.permute(0, 2, 1)
        residual1 = self.residual_conv1(x11)
        out_cnn1 = self.conv11(x11)
        out_cnn1 = self.relu1(out_cnn1)
        out_cnn1 = self.conv12(out_cnn1)
        out_cnn1 = self.relu1(out_cnn1)
        out_cnn1 = out_cnn1 + residual1
        out_cnn1 = self.relu1(out_cnn1)

        residual2 = self.residual_conv2(out_cnn1)
        out_cnn2 = self.conv13(out_cnn1)
        out_cnn2 = self.relu1(out_cnn2)
        out_cnn2 = self.conv14(out_cnn2)
        out_cnn2 = self.relu1(out_cnn2)
        out_cnn2 = out_cnn2 + residual2
        out_cnn2 = self.relu1(out_cnn2)

        out_cnn3 = out_cnn2.permute(0, 2, 1)
        out_cnn3 = self.dropout1(out_cnn3)
        out_cnn3 = torch.mean(out_cnn3, dim=1)
        out = self.mlp(out_cnn3)
        return out
