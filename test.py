import os

import cv2
import torch
import numpy as np
from natsort import natsorted

import utils.image_processing as imgproc
from utils.quality_assessment import NIQE
from networks.models import Generator
import config


def main() -> None:
    # Initialize the super-resolution model
    model = Generator(config.in_channels, config.out_channels, config.upscale_factor)
    model = model.to(device=config.device, memory_format=torch.channels_last)
    print("Build StarSRGAN model successfully.")

    # Load the super-resolution model weights
    checkpoint = torch.load(config.model_path, map_location=lambda storage, loc: storage)
    model_state_dict = model.state_dict()
    state_dict = {k.replace("model.", ""): v for k, v in checkpoint["ema_state_dict"].items() if
                  k.replace("model.", "") in model_state_dict.keys()}
    model.load_state_dict(state_dict)
    print(f"Load StarSRGAN model weights `{os.path.abspath(config.model_path)}` successfully.")

    # Create a folder of super-resolution experiment results
    if not os.path.exists(config.sr_dir):
        os.makedirs(config.sr_dir)

    # Start the verification mode of the model.
    model.eval()

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 1
    timings=np.zeros((repetitions,1))
    #GPU-WARM-UP
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float).to(config.device)
    for _ in range(10):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(mean_syn)
    print(std_syn)

    # Initialize the sharpness evaluation function
    niqe = NIQE(config.upscale_factor, config.niqe_model_path)

    # Set the sharpness evaluation function calculation device to the specified model
    niqe = niqe.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)

    # Initialize IQA metrics
    niqe_metrics = 0.0

    # Get a list of test image file names.
    file_names = natsorted(os.listdir(config.lr_dir))
    # Get the number of test image files.
    total_files = len(file_names)

    for index in range(total_files):
        lr_image_path = os.path.join(config.lr_dir, file_names[index])
        sr_image_path = os.path.join(config.sr_dir, file_names[index])

        print(f"Processing `{os.path.abspath(lr_image_path)}`...")
        # Read LR image
        lr_image = cv2.imread(lr_image_path, cv2.IMREAD_UNCHANGED)

        # Convert BGR channel image format data to RGB channel image format data
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

        # Convert RGB channel image format data to Tensor channel image format data
        lr_tensor = imgproc.image_to_tensor(lr_image, False, False).unsqueeze_(0)

        # Transfer Tensor channel image format data to CUDA device
        lr_tensor = lr_tensor.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)

        # Only reconstruct the Y channel image data.
        with torch.no_grad():
            sr_tensor = model(lr_tensor)

        # Save image
        sr_image = imgproc.tensor_to_image(sr_tensor, False, False)
        sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(sr_image_path, sr_image)

        # Cal IQA metrics
        niqe_metrics += niqe(sr_tensor).item()

    # Calculate the average value of the sharpness evaluation index,
    # and all index range values are cut according to the following values
    # NIQE range value is 0~100
    avg_niqe = 100 if niqe_metrics / total_files > 100 else niqe_metrics / total_files

    print(f"NIQE: {avg_niqe:4.2f} 100u")


if __name__ == "__main__":
    main()
