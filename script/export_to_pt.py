import argparse
import os

import torch
from torchvision.models.detection import fcos_resnet50_fpn
from torchvision.models.detection import FCOS_ResNet50_FPN_Weights


ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', type=str, default='models',
                help='Path to save the exported model')
args = vars(ap.parse_args())

# Load pretrained FCOS model
model = fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.DEFAULT)
model.eval()

# Create a dummy input for tracing (1 image of size 3x374x1238)
dummy_input = torch.randn(1, 3, 374, 1238)

# Convert to TorchScript (Tracing won't work with dynamic model logic)
scripted_model = torch.jit.script(model)
output_path = os.path.join(args['output'], 'fcos_resnet50_fpn_374x1238.pt')
scripted_model.save(output_path)
print("Model successfully scripted and saved as 'fcos_resnet50_fpn_374x1238.pt'")
