import matplotlib.pyplot as plt
import torch
import torchvision
from custom_datasets import afhqDataset
from models import UnetGenerator
from torch import nn
from torch.utils.data import DataLoader


def test(generator, val_dl, epoch, device):
    input_img, target_img, label = next(iter(val_dl))
    inputs = input_img.to(device)
    label = label.type(torch.int32).to(device)
    generated_output = generator((inputs, label))
    fig = plt.figure(figsize=(10, 2))

    labels = {0: "Cat", 1: "Dog", 2: "Wild"}
    for i in range(batch_size):
        fig.add_subplot(2,batch_size, i + 1)
        plt.imshow(input_img[i].cpu().detach().permute(1, 2, 0))
        plt.axis("off")
        plt.title(labels[label.cpu().detach().numpy()[i]])
        fig.add_subplot(2, batch_size, i + 11)
        plt.imshow(generated_output[i].cpu().detach().permute(1, 2, 0))
        plt.axis("off")

    plt.show()
    plt.savefig(f"results/test_{epoch}.png")


def save_images(images, path, nrow=8, normalize=True):
    if normalize:
        images = images.mul(0.5).add(0.5)
    grid = torchvision.utils.make_grid(images, nrow=nrow, padding=2)
    torchvision.utils.save_image(grid, path)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 10
    train_ds = afhqDataset(split="val")
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)

    generator = (
        UnetGenerator(3, 3, 64, norm_layer=nn.BatchNorm2d, use_dropout=False)
        .cuda()
        .float()
    )

    checkpoint = torch.load("checkpoints_pixandtext2/models_5.pth")
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()
    epoch = checkpoint["epoch"]
    test(generator, train_dl, epoch, device)
