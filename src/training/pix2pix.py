import glob

import torch
import torchvision
from custom_datasets import afhqDataset
from models import Discriminator, UnetGenerator
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm


# custom weights initialization called on generator and discriminator
def init_weights(net, init_type="normal", scaling=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv")) != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, scaling)
        elif (
            classname.find("BatchNorm2d") != -1
        ):  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            torch.nn.init.normal_(m.weight.data, 1.0, scaling)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def generator_loss(generated_image, target_img, G, real_target):
    adversarial_loss = nn.BCELoss()
    l1_loss = nn.L1Loss()

    gen_loss = adversarial_loss(G, real_target)
    l1_l = l1_loss(generated_image, target_img)
    gen_total_loss = gen_loss + (100 * l1_l)
    return gen_total_loss


def discriminator_loss(output, label):
    adversarial_loss = nn.BCELoss()
    disc_loss = adversarial_loss(output, label)
    return disc_loss


def train(
    num_epochs,
    generator,
    discriminator,
    train_dl,
    G_optimizer,
    D_optimizer,
    last_epoch=1,
):
    D_loss_plot, G_loss_plot = [], []
    for epoch in tqdm(range(last_epoch, num_epochs + 1)):
        D_loss_list, G_loss_list = [], []

        for (input_img, target_img, label) in tqdm(train_dl):

            D_optimizer.zero_grad()
            input_img = input_img.to(device)
            target_img = target_img.to(device)
            label = label.type(torch.int32).to(device)

            # ground truth labels real and fake
            real_target = Variable(torch.ones(input_img.size(0), 1, 30, 30).to(device))
            fake_target = Variable(torch.zeros(input_img.size(0), 1, 30, 30).to(device))

            # generator forward pass
            generated_image = generator((input_img, label))

            # train discriminator with fake/generated images
            disc_inp_fake = torch.cat((input_img, generated_image), 1)

            D_fake = discriminator((disc_inp_fake.detach(), label))

            D_fake_loss = discriminator_loss(D_fake, fake_target)

            # train discriminator with real images
            disc_inp_real = torch.cat((input_img, target_img), 1)

            D_real = discriminator((disc_inp_real,label))
            D_real_loss = discriminator_loss(D_real, real_target)

            # average discriminator loss
            D_total_loss = (D_real_loss + D_fake_loss) / 2
            D_loss_list.append(D_total_loss)
            # compute gradients and run optimizer step
            D_total_loss.backward()
            D_optimizer.step()

            # Train generator with real labels
            G_optimizer.zero_grad()
            fake_gen = torch.cat((input_img, generated_image), 1)
            G = discriminator((fake_gen, label))
            G_loss = generator_loss(generated_image, target_img, G, real_target)
            G_loss_list.append(G_loss)
            # compute gradients and run optimizer step
            torch.autograd.set_detect_anomaly(True)
            G_loss.backward()
            G_optimizer.step()

        # Save checkpoints
        if epoch % 5 == 0:
            print("saving model")
            # test(generator, train_dl, epoch, device)  # TODO change to valid_dl
            torch.save(
                {
                    "epoch": epoch,
                    "generator_state_dict": generator.state_dict(),
                    "G_optimizer_state_dict": G_optimizer.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "D_optimizer_state_dict": D_optimizer.state_dict(),
                    "G_loss": G_loss,
                    "D_loss": D_total_loss,
                },
                f"checkpoints_pixandtext/models_{epoch}.pth",
            )

    return generator, discriminator


def test(generator, val_dl, epoch, device):
    for input_img, target_img, label in val_dl:
        inputs = input_img.to(device)
        generated_output = generator(inputs)
        save_images(
            generated_output.data[:10],
            "results/sample_%d" % epoch + ".png",
            nrow=5,
            normalize=True,
        )


def save_images(images, path, nrow=8, normalize=True):
    if normalize:
        images = images.mul(0.5).add(0.5)
    grid = torchvision.utils.make_grid(images, nrow=nrow, padding=2)
    torchvision.utils.save_image(grid, path)


if __name__ == "__main__":
    batch_size = 32

    train_ds = afhqDataset()
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA device count:", torch.cuda.device_count())
    # torch.cuda.empty_cache()
    generator = (
        UnetGenerator(3, 3, 64, norm_layer=nn.BatchNorm2d, use_dropout=False)
        .cuda()
        .float()
    )
    init_weights(generator, "normal", scaling=0.02)

    discriminator = Discriminator(6, 64, norm_layer=nn.BatchNorm2d).cuda().float()
    init_weights(discriminator, "normal", scaling=0.02)

    G_optimizer = torch.optim.Adam(
        generator.parameters(), lr=0.0002, betas=(0.5, 0.999)
    )
    D_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
    )
    epoch = 1
    resume = True
    if resume:
        print("loading model")
        checkpoints = sorted(glob.glob("checkpoints_pixandtext/*.pth"))
        print(checkpoints[-1])
        checkpoint = torch.load(checkpoints[-1])
        generator.load_state_dict(checkpoint["generator_state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        G_optimizer.load_state_dict(checkpoint["G_optimizer_state_dict"])
        D_optimizer.load_state_dict(checkpoint["D_optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        G_loss = checkpoint["G_loss"]
        D_loss = checkpoint["D_loss"]
        generator.eval()
        discriminator.eval()
    print(epoch)

    train(
        200,
        generator,
        discriminator,
        train_dl,
        G_optimizer,
        D_optimizer,
        last_epoch=epoch,
    )
