import argparse

import cv2
import numpy as np
import torch

import utils.image_processing as imgproc
from networks.models import Generator
import config


def main(args):
    # Initialize the model
    model = Generator(config.in_channels, config.out_channels, config.upscale_factor)
    model = model.to(memory_format=torch.channels_last, device=config.device)
    print("Build StarSRGAN model successfully.")

    # Load the CRNN model weights
    checkpoint = torch.load(
        args.weights_path, map_location=lambda storage, loc: storage
    )
    model.load_state_dict(
        {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
    )
    print(f"Load StarSRGAN model weights `{args.weights_path}` successfully.")

    # Start the verification mode of the model.
    model.eval()

    # Read LR image and HR image
    lr_image = (
        cv2.imread(args.inputs_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
    )

    # Convert BGR channel image format data to RGB channel image format data
    lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

    # Convert RGB channel image format data to Tensor channel image format data
    lr_tensor = imgproc.image_to_tensor(lr_image, False, False).unsqueeze_(0)

    # Transfer Tensor channel image format data to CUDA device
    lr_tensor = lr_tensor.to(
        device=config.device, memory_format=torch.channels_last, non_blocking=True
    )

    # Use the model to generate super-resolved images
    with torch.no_grad():
        sr_tensor = model(lr_tensor)

    # Save image
    sr_image = imgproc.tensor_to_image(sr_tensor, False, False)
    sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output_path, sr_image)

    print(f"SR image save to `{args.output_path}`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Using the StarSRGAN model generator super-resolution images."
    )
    parser.add_argument("--inputs_path", type=str, help="Low-resolution image path.")
    parser.add_argument("--output_path", type=str, help="Super-resolution image path.")
    parser.add_argument("--weights_path", type=str, help="Model weights file path.")
    args = parser.parse_args()

    main(args)
