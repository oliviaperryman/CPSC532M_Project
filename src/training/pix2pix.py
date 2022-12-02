import torch
import torchvision.transforms as transforms
import numpy as np
from custom_transforms import (
    normalize,
    random_crop,
    random_jittering_mirroring,
)
from models import Discriminator, UnetGenerator
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from custom_datasets import afhqDataset


# class Train(object):
#     def __call__(self, dl_data):
#         # 2 images processed by DL to resize to 286x286, and label
#         # image, contour, label = dl_data
#         # tar = np.array(image).astype(np.float32)
#         # inp = np.array(contour).astype(np.float32)
#         # inp, tar = random_jittering_mirroring(inp, tar)
#         # inp, tar = normalize(inp, tar)
#         # image_a = torch.from_numpy(inp.copy().transpose((2, 0, 1)))
#         # image_b = torch.from_numpy(tar.copy().transpose((2, 0, 1)))
#         # return image_a, image_b, label


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
    # print(gen_loss)
    return gen_total_loss


def discriminator_loss(output, label):
    adversarial_loss = nn.BCELoss()
    disc_loss = adversarial_loss(output, label)
    return disc_loss


def train(num_epochs, generator, discriminator, train_dl, G_optimizer, D_optimizer):
    D_loss_plot, G_loss_plot = [], []
    for epoch in range(1, num_epochs + 1):

        D_loss_list, G_loss_list = [], []

        for (input_img, target_img, label) in train_dl:

            D_optimizer.zero_grad()
            input_img = input_img.to(device)
            target_img = target_img.to(device)

            # ground truth labels real and fake
            real_target = Variable(torch.ones(input_img.size(0), 1, 30, 30).to(device))
            fake_target = Variable(torch.zeros(input_img.size(0), 1, 30, 30).to(device))

            print(input_img.shape)

            # generator forward pass
            generated_image = generator(input_img)

            # train discriminator with fake/generated images
            disc_inp_fake = torch.cat((input_img, generated_image), 1)

            D_fake = discriminator(disc_inp_fake.detach())

            D_fake_loss = discriminator_loss(D_fake, fake_target)

            # train discriminator with real images
            disc_inp_real = torch.cat((input_img, target_img), 1)

            D_real = discriminator(disc_inp_real)
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
            G = discriminator(fake_gen)
            G_loss = generator_loss(generated_image, target_img, G, real_target)
            G_loss_list.append(G_loss)
            # compute gradients and run optimizer step
            G_loss.backward()
            G_optimizer.step()
    return generator, discriminator


def test(val_dl, epoch):
    for (inputs, targets), _ in val_dl:
        inputs = inputs.to(device)
        generated_output = generator(inputs)
        save_images(
            generated_output.data[:10],
            "sample_%d" % epoch + ".png",
            nrow=5,
            normalize=True,
        )


if __name__ == "__main__":
    # DIR = "edges2shoes/train_data/"
    batch_size = 1

    # train_ds = ImageFolder(DIR, transform=transforms.Compose([Train()]))
    train_ds = afhqDataset()
    train_dl = DataLoader(train_ds, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.device_count())
    generator = (
        UnetGenerator(3, 3, 64, norm_layer=nn.BatchNorm2d, use_dropout=False)
        # .cuda()
        .cpu()
        .float()
    )
    init_weights(generator, "normal", scaling=0.02)
    generator = torch.nn.DataParallel(generator)  # multi-GPUs

    # The following things were not defined in the tutorial, so I guessed

    # discriminator = Discriminator(6, 64, norm_layer=nn.BatchNorm2d).cuda().float()
    discriminator = Discriminator(6, 64, norm_layer=nn.BatchNorm2d).cpu().float()

    G_optimizer = torch.optim.Adam(
        generator.parameters(), lr=0.0002, betas=(0.5, 0.999)
    )
    D_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
    )

    generator, discriminator = train(
        1, generator, discriminator, train_dl, G_optimizer, D_optimizer
    )

    test(train_dl, 1)
