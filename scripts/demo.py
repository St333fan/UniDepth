import numpy as np
import torch
from PIL import Image

from unidepth.models import UniDepthV1, UniDepthV2, UniDepthV2old
from unidepth.utils import colorize, image_grid
from unidepth.utils.camera import Pinhole


def demo(model):
    rgb = np.array(Image.open("assets/demo/PXL_20250430_135003725.jpg"))
    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
    intrinsics_torch = torch.from_numpy(np.load("assets/demo/intrinsics.npy"))
    print(intrinsics_torch)
    intrinsics_torch= torch.tensor([[2833,   0.0000, 2040.0000],
                                      [  0.0000, 2833, 1588.0000],
                                      [  0.0000,   0.0000,   1.0000]])
    camera = Pinhole(K=intrinsics_torch.unsqueeze(0))
    
    # infer method of V1 uses still the K matrix as input
    if isinstance(model, (UniDepthV2old, UniDepthV1)):
        camera = camera.K.squeeze(0)

    import matplotlib.pyplot as plt  # Import for adding text

# ... (previous code) ...

# predict
    predictions = model.infer(rgb_torch, camera)

# get predicted depth
    depth_pred = predictions["depth"].squeeze().cpu().numpy()

# colorize predicted depth
    depth_pred_col = colorize(depth_pred, vmin=0.01, vmax=10.0, cmap="magma_r")

   # Convert the colorized depth image to RGB 
    depth_pred_col_rgb = Image.fromarray(depth_pred_col.astype('uint8')).convert('RGB') 

# Create a figure and axes for the image
    fig, ax = plt.subplots(figsize=(10, 10))  # Adjust size as needed
    ax.imshow(depth_pred_col_rgb)

# Add depth values at specific locations (example)
    for x in range(0, depth_pred.shape[1], 500):  # Every 100 pixels horizontally
        for y in range(0, depth_pred.shape[0], 500):  # Every 100 pixels vertically
           depth_value = depth_pred[y, x]
           ax.text(x, y, f"{depth_value:.2f}", color='white', fontsize=8)

# Save the image with depth values
    plt.savefig("assets/demo/predicted_depth_with_values.png")
    plt.close(fig)  # Close the figure to release resources


if __name__ == "__main__":
    print("Torch version:", torch.__version__)
    type_ = "l"  # available types: s, b, l
    name = f"unidepth-v2-vit{type_}14"
    model = UniDepthV2.from_pretrained(f"lpiccinelli/{name}")

    # set resolution level (only V2)
    # model.resolution_level = 9

    # set interpolation mode (only V2)
    model.interpolation_mode = "bilinear"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    demo(model)
