#!/usr/bin/env python3
"""Script to export a pre-trained FCOS model to TorchScript format."""

import argparse
import os

import torch
from torchvision.models.detection import fcos_resnet50_fpn
from torchvision.models.detection import FCOS_ResNet50_FPN_Weights


def export_fcos_model(output_path, input_height, input_width):
    """Export FCOS model to TorchScript format."""
    print('Creating FCOS model...')
    # Load pretrained FCOS model
    model = fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.DEFAULT)
    model.eval()

    print('Exporting to TorchScript...')
    try:
        # Use scripting for FCOS as it contains dynamic logic that tracing can't handle
        scripted_model = torch.jit.script(model)
        scripted_model.save(output_path)
        print(f'TorchScript model saved to: {output_path}')
    except Exception as e:
        print(f'✗ TorchScript export failed: {e}')
        raise

    # Test the exported model
    print('\nTesting TorchScript model...')
    try:
        loaded_model = torch.jit.load(output_path)
        print('Preparing dummy input...')
        dummy_input = [torch.randn(3, input_height, input_width)]
        test_output = loaded_model(dummy_input)
        print(f'✓ TorchScript model validation passed - Output type: {type(test_output)}')
        if isinstance(test_output, list) and len(test_output) > 0:
            print(f'  Number of detections: {len(test_output)}')
            if 'boxes' in test_output[0]:
                print(f'  Boxes shape: {test_output[0]["boxes"].shape}')
            if 'scores' in test_output[0]:
                print(f'  Scores shape: {test_output[0]["scores"].shape}')
            if 'labels' in test_output[0]:
                print(f'  Labels shape: {test_output[0]["labels"].shape}')
    except Exception as e:
        print(f'✗ TorchScript model validation failed: {e}')


if __name__ == '__main__':
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('--height', type=int, default=374, help='The height of the input image')
    ap.add_argument('--width', type=int, default=1238, help='The width of the input image')
    ap.add_argument('--output-dir', type=str, default='models',
                    help='The path to output .pt file')
    args = vars(ap.parse_args())

    # Create output directory if it doesn't exist
    os.makedirs(args['output_dir'], exist_ok=True)

    height = args['height']
    width = args['width']
    output_dir = args['output_dir']

    # Export to TorchScript
    print(f'=== Exporting FCOS with ResNet50 FPN backbone for input size: {height}x{width} ===')
    output_path = os.path.join(output_dir, f'fcos_resnet50_fpn_{height}x{width}.pt')
    export_fcos_model(output_path=output_path, input_width=width, input_height=height)

    print('TorchScript export completed.')
