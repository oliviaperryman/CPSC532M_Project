import torch
import torchvision
from torch import nn

from custom_datasets import afhqDataset
from models import UnetGenerator
from torch.utils.data import DataLoader

def test(generator, val_dl, epoch, device):
    for input_img, target_img, label in val_dl:
        inputs = input_img.to(device)
        generated_output = generator(inputs)
        save_images(
            generated_output.data[:10],
            "sample_%d" % epoch + ".png",
            nrow=5,
            normalize=True,
        )

def save_images(images, path, nrow=8, normalize=True):
    if normalize:
        images = images.mul(0.5).add(0.5)
    grid = torchvision.utils.make_grid(images, nrow=nrow, padding=2)
    torchvision.utils.save_image(grid, path)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 10
    train_ds = afhqDataset()
    train_dl = DataLoader(train_ds, batch_size)
    generator = (
        UnetGenerator(3, 3, 64, norm_layer=nn.BatchNorm2d, use_dropout=False)
        .cuda()
        .float()
    )

    checkpoint = torch.load("checkpoints/models_10.pth")
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()
    test(generator, train_dl, 1, device)
