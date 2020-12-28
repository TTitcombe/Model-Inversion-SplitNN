from typing import Optional

import matplotlib.pyplot as plt
import torch
import torchvision


def plot_images(tensors, rows: Optional[int] = None, savepath: Optional[str] = None):
    """
    Plot normalised MNIST tensors as images
    """
    fig = plt.figure(figsize=(20, 10))

    n_tensors = len(tensors)

    # De-normalise an MNIST tensor
    mu = torch.tensor([0.1307], dtype=torch.float32)
    sigma = torch.tensor([0.3081], dtype=torch.float32)
    Unnormalise = torchvision.transforms.Normalize(
        (-mu / sigma).tolist(), (1.0 / sigma).tolist()
    )

    images = []
    for tensor in tensors:
        tensor = Unnormalise(tensor)

        # Clip image values so we can plot
        tensor[tensor < 0] = 0
        tensor[tensor > 1] = 1

        tensor = tensor.unsqueeze(0)  # add batch dim
        images.append(tensor)

    images = torch.cat(images)
    grid_image = torchvision.utils.make_grid(images, nrow=rows).permute(1, 2, 0)

    if savepath:
        plt.imsave(savepath, grid_image.detach().numpy())
    else:
        plt.imshow(grid_image)
