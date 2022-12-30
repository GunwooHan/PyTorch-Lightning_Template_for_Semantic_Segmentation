import math

import torch

def train_test_split(X, y, test_size=0.2, random_state=42):
    n = len(X)
    n_test = int(n * test_size)
    n_train = n - n_test
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]

@torch.no_grad()
def kaiming_init(model):
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, 0, 0.01)
            torch.nn.init.constant_(m.bias, 0)

