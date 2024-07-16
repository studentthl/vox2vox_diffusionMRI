import argparse
import os
import numpy as np
import time
import datetime
import sys
import matplotlib.pyplot as plt
import json

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

from model import *
from dataset2 import *
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def load_checkpoint(filename):
    return torch.load(filename)

def save_losses(g_losses, d_losses, adv_losses, g_accuracies, d_accuracies, filename="losses.json"):
    with open(filename, 'w') as f:
        json.dump({
            'g_losses': g_losses, 
            'd_losses': d_losses, 
            'adv_losses': adv_losses, 
            'g_accuracies': g_accuracies, 
            'd_accuracies': d_accuracies
        }, f)

def load_losses(filename="losses.json"):
    with open(filename, 'r') as f:
        losses = json.load(f)
    return losses['g_losses'], losses['d_losses'], losses['adv_losses'], losses['g_accuracies'], losses['d_accuracies']

def train():

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=5, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="qspace_3d", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--glr", type=float, default=0.0002, help="adam: generator learning rate")
    parser.add_argument("--dlr", type=float, default=0.0002, help="adam: discriminator learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=33, help="size of image height")
    parser.add_argument("--img_width", type=int, default=33, help="size of image width")
    parser.add_argument("--img_depth", type=int, default=33, help="size of image depth")
    parser.add_argument("--channels", type=int, default=2, help="number of image channels")
    parser.add_argument("--disc_update", type=int, default=5, help="only update discriminator every n iter")
    parser.add_argument("--d_threshold", type=float, default=0.8, help="discriminator threshold")
    parser.add_argument("--threshold", type=int, default=-1, help="threshold during sampling, -1: No thresholding")
    parser.add_argument("--sample_interval", type=int, default=1, help="interval between sampling of images from generators")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
    parser.add_argument("--checkpoint_path", type=str, default="saved_models/checkpoint.pth", help="path to save checkpoints")
    parser.add_argument("--losses_path", type=str, default="saved_models/losses.json", help="path to save losses")

    opt = parser.parse_args()
    print(opt)

    os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
    os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

    cuda = True if torch.cuda.is_available() else False

    # Loss functions
    criterion_GAN = nn.MSELoss()
    criterion_voxelwise = nn.MSELoss()

    # Loss weight of L1 voxel-wise loss between translated image and real image
    lambda_voxel = 100

    # Calculate output of image discriminator (PatchGAN)
    patch = (1, 3, 3, 3)

    # Initialize generator and discriminator
    generator = GeneratorUNet(in_channels=opt.channels, out_channels=opt.channels)
    discriminator = Discriminator(in_channels=opt.channels)

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        criterion_GAN.cuda()
        criterion_voxelwise.cuda()

    if opt.epoch != 0:
        # Load pretrained models
        generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))
        discriminator.load_state_dict(torch.load("saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, opt.epoch)))
        checkpoint = load_checkpoint(opt.checkpoint_path)
        g_losses, d_losses, adv_losses, g_accuracies, d_accuracies = load_losses(opt.losses_path)
        start_epoch = checkpoint['epoch'] + 1
    else:
        # Initialize weights
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)
        start_epoch = 0
        g_losses = []
        d_losses = []
        adv_losses = []
        g_accuracies = []
        d_accuracies = []

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.glr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.dlr, betas=(opt.b1, opt.b2))

    # Configure dataloaders
    transforms_ = transforms.Compose([
        transforms.ToTensor()
    ])

    # Configure dataloaders
    file_paths = [f'Data/simulation_results_{i:02d}.mat' for i in range(1, 21)]
    train_dataset, val_dataset = get_train_val_datasets(file_paths)

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # Tensor type
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def sample_voxel_volumes(epoch):
        imgs = next(iter(val_loader))
        real_A = Variable(imgs["Undersampled_qspace"].type(Tensor))
        real_B = Variable(imgs["Q_space"].type(Tensor))

        real_A = real_A.permute(0, 4, 1, 2, 3)
        real_B = real_B.permute(0, 4, 1, 2, 3)

        fake_B = generator(real_A)

        real_A = real_A.cpu().detach().numpy()
        real_B = real_B.cpu().detach().numpy()
        fake_B = fake_B.cpu().detach().numpy()

        image_folder = "images/%s/epoch_%s_" % (opt.dataset_name, epoch)

        hf = h5py.File(image_folder + 'real_undersampled_A.vox', 'w')
        hf.create_dataset('data', data=real_A)

        hf1 = h5py.File(image_folder + 'real_qspace_B.vox', 'w')
        hf1.create_dataset('data', data=real_B)

        hf2 = h5py.File(image_folder + 'fake_qspace_B.vox', 'w')
        hf2.create_dataset('data', data=fake_B)

    prev_time = time.time()
    discriminator_update = False
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(train_loader):

            # Model inputs
            real_A = Variable(batch["Undersampled_qspace"].type(Tensor))
            real_B = Variable(batch["Q_space"].type(Tensor))

            real_A = real_A.permute(0, 4, 1, 2, 3)
            real_B = real_B.permute(0, 4, 1, 2, 3)

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

            # Train Discriminator
            fake_B = generator(real_A)
            pred_real = discriminator(real_B, real_B)
            loss_real = criterion_GAN(pred_real, valid)

            pred_fake = discriminator(fake_B.detach(), real_B)
            loss_fake = criterion_GAN(pred_fake, fake)
            loss_D = 0.5 * (loss_real + loss_fake)

            d_real_acu = torch.ge(pred_real.squeeze(), 0.5).float()
            d_fake_acu = torch.le(pred_fake.squeeze(), 0.5).float()
            d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_acu), 0))

            d_accuracies.append(d_total_acu.item())

            if d_total_acu <= opt.d_threshold:
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()
                discriminator_update = True

            # Train Generator
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()

            fake_B = generator(real_A)
            pred_fake = discriminator(fake_B, real_B)
            loss_GAN = criterion_GAN(pred_fake, valid)
            loss_voxel = criterion_voxelwise(fake_B, real_B)

            loss_G = loss_GAN + lambda_voxel * loss_voxel

            # Calculate and store adversarial loss and G accuracy
            adv_losses.append(loss_GAN.item())
            g_acu = torch.mean(torch.ge(pred_fake.squeeze(), 0.5).float())
            g_accuracies.append(g_acu.item())

            loss_G.backward()
            optimizer_G.step()

            g_losses.append(loss_G.item())
            d_losses.append(loss_D.item())

            batches_done = epoch * len(train_loader) + i

            # Log Progress
            batches_left = opt.n_epochs * len(train_loader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f, D accuracy: %f, D update: %s] [G loss: %f, voxel: %f, adv: %f, G accuracy: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(train_loader),
                    loss_D.item(),
                    d_total_acu,
                    str(discriminator_update),
                    loss_G.item(),
                    loss_voxel.item(),
                    loss_GAN.item(),
                    g_acu.item(),
                    time_left,
                )
            )
            if batches_done % (opt.sample_interval * len(train_loader)) == 0:
                sample_voxel_volumes(epoch)
                print('*****volumes sampled*****')

            discriminator_update = False

            # Save the losses and accuracies after each batch
            save_losses(g_losses, d_losses, adv_losses, g_accuracies, d_accuracies, filename=opt.losses_path)

        # Save the losses and accuracies after each epoch
        save_losses(g_losses, d_losses, adv_losses, g_accuracies, d_accuracies, filename=opt.losses_path)

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch))
            torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, epoch))
            save_checkpoint({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict()
            }, filename=opt.checkpoint_path)
            save_losses(g_losses, d_losses, adv_losses, g_accuracies, d_accuracies, filename=opt.losses_path)

    # Save final models
    torch.save(generator.state_dict(), "saved_models/%s/generator_final.pth" % opt.dataset_name)
    torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_final.pth" % opt.dataset_name)
    save_checkpoint({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict()
    }, filename="saved_models/%s/final_checkpoint.pth" % opt.dataset_name)
    save_losses(g_losses, d_losses, adv_losses, g_accuracies, d_accuracies, filename="saved_models/%s/final_losses.json" % opt.dataset_name)

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    train()


