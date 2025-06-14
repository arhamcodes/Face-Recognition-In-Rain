import torch
from PIL import Image
from torchvision import transforms
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataLoader():
    def __init__(self, dataset_name="/content/rain", img_res=(256,256)):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.n_batches = 0
        self.transform = transforms.Compose([
            transforms.Resize(self.img_res),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # def load_data(self, batch_size=1, is_testing=False):
    def load_data(self, batch_size=1, is_testing=True):
        from glob import glob
        import numpy as np
        data_type = "training" if not is_testing else "test_nature"
        # data_type = "training" if not is_testing else "test"
        # data_type = "train" if not is_testing else "test"
        path = glob('%s/%s/*' % (self.dataset_name, data_type))
        # path = glob('%s/%s/*' % (self.dataset_name, data_type))[:700]
        batch_images = np.random.choice(path, size=batch_size)
        imgs_A, imgs_B = [], []

        for img_path in batch_images:
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            half_w = w // 2
            img_A = img.crop((0, 0, half_w, h))
            img_B = img.crop((half_w, 0, w, h))

            if not is_testing and random.random() < 0.5:
                img_A = transforms.functional.hflip(img_A)
                img_B = transforms.functional.hflip(img_B)

            img_A = self.transform(img_A)  # shape [3, H, W]
            img_B = self.transform(img_B)
            imgs_A.append(img_A)
            imgs_B.append(img_B)

        imgs_A = torch.stack(imgs_A, dim=0)  # shape [batch, 3, H, W]
        imgs_B = torch.stack(imgs_B, dim=0)
        return imgs_A, imgs_B

    # def load_batch(self, batch_size=1, is_testing=False):
    def load_batch(self, batch_size=1, is_testing=True):
        from glob import glob
        import numpy as np
        # data_type = "training" if not is_testing else "test_syn"
        data_type = "training" if not is_testing else "test_nature"
        # data_type = "training" if not is_testing else "test"
        # data_type = "train" if not is_testing else "test"
        path = glob('%s/%s/*' % (self.dataset_name, data_type))
        # path = glob('%s/%s/*' % (self.dataset_name, data_type))[:700]
        self.n_batches = int(len(path) / batch_size)

        for i in range(self.n_batches - 1):
            batch = path[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img_path in batch:
                img = Image.open(img_path).convert('RGB')
                w, h = img.size
                half_w = w // 2
                img_A = img.crop((0, 0, half_w, h))
                img_B = img.crop((half_w, 0, w, h))

                if not is_testing and random.random() > 0.5:
                    img_A = transforms.functional.hflip(img_A)
                    img_B = transforms.functional.hflip(img_B)

                img_A = self.transform(img_A)
                img_B = self.transform(img_B)
                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = torch.stack(imgs_A, dim=0)
            imgs_B = torch.stack(imgs_B, dim=0)
            yield imgs_A, imgs_B

    def imread(self, path):
        # Not really used in this PyTorch version, but kept for reference
        return Image.open(path).convert('RGB')
# ...existing code...

# Commented out IPython magic to ensure Python compatibility.
# ...existing code...

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import datetime
import os
# import matplotlib.pyplot as plt
from glob import glob
# from data_loader import DataLoader

class Discriminator(nn.Module):
    def __init__(self, in_channels=6):
        super(Discriminator, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )
        # Simple pyramid pool replacement
        self.conv_out = nn.Sequential(
            nn.Conv2d(64, 72, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, imgA, imgB):
        imgA = F.interpolate(imgA, size=(256, 256))
        imgB = F.interpolate(imgB, size=(256, 256))
        x = torch.cat((imgA, imgB), dim=1)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.conv_out(x)
        return x

# class DenseBlock(nn.Module):
#     def __init__(self, num_layers, growth):
#         super(DenseBlock, self).__init__()
#         self.layers = nn.ModuleList()
#         for _ in range(num_layers):
#             self.layers.append(nn.Sequential(
#                 nn.Conv2d(growth, growth, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(growth),
#                 nn.LeakyReLU(0.2, inplace=True)
#             ))
#         self.growth = growth

#     def forward(self, x):
#         features = [x]
#         for layer in self.layers:
#             out = layer(x)
#             x = torch.cat([x, out], dim=1)
#         return x

class DenseBlock(nn.Module):
    def __init__(self, num_layers, growth):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.growth = growth
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(growth * (i + 1), growth, kernel_size=3, padding=1),
                nn.BatchNorm2d(growth),
                nn.LeakyReLU(0.2, inplace=True)
            ))

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)

# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.down1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2)
#         )
#         self.dense1 = DenseBlock(num_layers=4, growth=64)
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(64+256, 128, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True)
#         )
#         self.dense2 = DenseBlock(num_layers=6, growth=128)
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(128+768, 256, kernel_size=3, stride=2),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True)
#         )
#         self.dense3 = DenseBlock(num_layers=8, growth=256)
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(256+2048, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True)
#         )
#         self.dense4 = DenseBlock(num_layers=8, growth=512)
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(512+4096, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True)
#         )
#         self.up1 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             nn.Conv2d(128, 120, kernel_size=4, stride=1, padding=1),
#             nn.Dropout(0.0),
#             nn.BatchNorm2d(120),
#             nn.ReLU(inplace=True)
#         )
#         self.dense5 = DenseBlock(num_layers=6, growth=120)
#         self.up2 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             nn.Conv2d(120+720, 64, kernel_size=4, stride=1, padding=1),
#             nn.Dropout(0.0),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True)
#         )
#         self.dense6 = DenseBlock(num_layers=4, growth=64)
#         self.up3 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             nn.Conv2d(64+256, 64, kernel_size=4, stride=1, padding=1),
#             nn.Dropout(0.0),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True)
#         )
#         self.dense7 = DenseBlock(num_layers=4, growth=64)
#         self.conv5 = nn.Sequential(
#             nn.Conv2d(64+256, 16, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(inplace=True)
#         )
#         self.pad = nn.ReflectionPad2d(5)
#         self.conv6 = nn.Conv2d(16, 3, kernel_size=3)
#         self.out = nn.Tanh()

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.dense1 = DenseBlock(num_layers=4, growth=64)
        self.conv1 = nn.Sequential(
            nn.Conv2d(64+256, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dense2 = DenseBlock(num_layers=6, growth=128)
        self.conv2 = nn.Sequential(
            nn.Conv2d(128+768, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.dense3 = DenseBlock(num_layers=8, growth=256)
        self.conv3 = nn.Sequential(
            nn.Conv2d(256+2048, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.dense4 = DenseBlock(num_layers=8, growth=512)
        self.conv4 = nn.Sequential(
            nn.Conv2d(512+4096, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 120, kernel_size=4, stride=1, padding=1),
            nn.Dropout(0.0),
            nn.BatchNorm2d(120),
            nn.ReLU(inplace=True)
        )
        self.dense5 = DenseBlock(num_layers=6, growth=120)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(120+720, 64, kernel_size=4, stride=1, padding=1),
            nn.Dropout(0.0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dense6 = DenseBlock(num_layers=4, growth=64)
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64+256, 64, kernel_size=4, stride=1, padding=1),
            nn.Dropout(0.0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dense7 = DenseBlock(num_layers=4, growth=64)
        self.conv5 = nn.Sequential(
            nn.Conv2d(64+256, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.pad = nn.ReflectionPad2d(4)  # Adjust padding to ensure output size matches input size
        self.conv6 = nn.Conv2d(16, 3, kernel_size=3, padding=1)  # Adjust padding to ensure output size matches input size
        self.out = nn.Tanh()

    def forward(self, x):
        x0 = self.down1(x)
        db1 = self.dense1(x0)
        c1 = self.conv1(db1)
        db2 = self.dense2(c1)
        c2 = self.conv2(db2)
        db3 = self.dense3(c2)
        c3 = self.conv3(db3)
        db4 = self.dense4(c3)
        c4 = self.conv4(db4)
        u1 = self.up1(c4)
        db5 = self.dense5(u1)
        u2 = self.up2(db5)
        db6 = self.dense6(u2)
        u3 = self.up3(db6)
        db7 = self.dense7(u3)
        c5 = self.conv5(db7)
        c5 = self.pad(c5)
        c6 = self.conv6(c5)
        return self.out(c6)

# class IDGAN:
#     def __init__(self):
#         self.img_rows = 256
#         self.img_cols = 256
#         self.channels = 3
#         self.img_shape = (self.channels, self.img_rows, self.img_cols)
#         self.dataset_name = 'rain'
#         self.data_loader = DataLoader(dataset_name=self.dataset_name, img_res=(self.img_rows, self.img_cols))
#         self.disc_out = (72, 14, 14)
#         self.generator = Generator().cuda()
#         self.discriminator = Discriminator().cuda()
#         self.optim_g = optim.Adam(self.generator.parameters(), lr=1e-3, betas=(0.9, 0.999))
#         self.optim_d = optim.SGD(self.discriminator.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-6, nesterov=False)
#         self.adversarial_criterion = nn.MSELoss()
#         self.l1_criterion = nn.L1Loss()

#     def train(self, epochs, batch_size, sample_interval=28):
#         valid = torch.ones(batch_size, *self.disc_out).cuda()
#         fake = torch.zeros(batch_size, *self.disc_out).cuda()
#         start_time = datetime.datetime.now()

#         for epoch in range(epochs):
#             for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):
#                 # print(imgs_A.shape, imgs_B.shape)
#                 real_A = torch.FloatTensor(imgs_A).cuda()
#                 real_B = torch.FloatTensor(imgs_B).cuda()

#                 # Train Discriminator
#                 self.discriminator.train()
#                 self.optim_d.zero_grad()
#                 fake_A = self.generator(real_B)
#                 fake_A = F.interpolate(fake_A, size=(256, 256))
#                 pred_real = self.discriminator(real_A, real_B)
#                 loss_real = self.adversarial_criterion(pred_real, valid)
#                 pred_fake = self.discriminator(fake_A.detach(), real_B)
#                 loss_fake = self.adversarial_criterion(pred_fake, fake)
#                 d_loss = 0.5 * (loss_real + loss_fake)
#                 d_loss.backward()
#                 self.optim_d.step()

#                 # Train Generator
#                 self.generator.train()
#                 self.optim_g.zero_grad()
#                 pred_fake = self.discriminator(fake_A, real_B)
#                 g_loss_adv = self.adversarial_criterion(pred_fake, valid)
#                 l1_loss = self.l1_criterion(fake_A, real_A)
#                 g_loss = g_loss_adv + 10.0 * l1_loss
#                 g_loss.backward()
#                 self.optim_g.step()

#                 elapsed_time = datetime.datetime.now() - start_time
#                 print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] time: %s"
#                       % (epoch, epochs, batch_i, self.data_loader.n_batches, d_loss.item(), g_loss.item(), elapsed_time))

#                 if batch_i % sample_interval == 0:
#                     self.sample_images(epoch, batch_i, real_A, fake_A)

#     # def sample_images(self, epoch, batch_i, real_A, fake_A):
#     #     os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
#     #     r, c = 3, 3
#     #     real_A_np = real_A.permute(0,2,3,1).cpu().data.numpy()
#     #     fake_A_np = fake_A.permute(0,2,3,1).cpu().data.numpy()
#     #     fig, axs = plt.subplots(r, c)
#     #     gen_imgs = np.concatenate([fake_A_np, real_A_np, real_A_np])
#     #     gen_imgs = 0.5 * gen_imgs + 0.5
#     #     titles = ['WithRain', 'Generated', 'Original']
#     #     cnt = 0
#     #     for i in range(r):
#     #         for j in range(c):
#     #             axs[i,j].imshow(gen_imgs[cnt])
#     #             axs[i,j].set_title(titles[i])
#     #             axs[i,j].axis('off')
#     #             cnt += 1
#     #     fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
#     #     plt.close()

#     def sample_images(self, epoch, batch_i, real_A, fake_A):
#         os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
#         r, c = 1, 3  # Adjust grid size to 1x3
#         real_A_np = real_A.permute(0, 2, 3, 1).cpu().data.numpy()
#         fake_A_np = fake_A.permute(0, 2, 3, 1).cpu().data.numpy()
#         fig, axs = plt.subplots(r, c)
#         gen_imgs = np.concatenate([real_A_np, fake_A_np, real_A_np])
#         gen_imgs = 0.5 * gen_imgs + 0.5
#         titles = ['Original', 'Generated', 'Original']
#         cnt = 0
#         for i in range(r):
#             for j in range(c):
#                 axs[j].imshow(gen_imgs[cnt])  # Adjust indexing to axs[j]
#                 axs[j].set_title(titles[j])  # Adjust indexing to titles[j]
#                 axs[j].axis('off')
#                 cnt += 1
#         fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
#         plt.close()

# gan = IDGAN()
# gan.train(epochs=150, batch_size=8, sample_interval=25)

#######################################################################################

# class IDGAN:
#     def __init__(self):
#         self.img_rows = 256
#         self.img_cols = 256
#         self.channels = 3
#         self.img_shape = (self.channels, self.img_rows, self.img_cols)
#         self.dataset_name = 'rain'
#         self.data_loader = DataLoader(dataset_name=self.dataset_name, img_res=(self.img_rows, self.img_cols))
#         self.disc_out = (72, 14, 14)
#         self.generator = Generator().cuda()
#         self.discriminator = Discriminator().cuda()
#         self.optim_g = optim.Adam(self.generator.parameters(), lr=1e-3, betas=(0.9, 0.999))
#         self.optim_d = optim.SGD(self.discriminator.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-6, nesterov=False)
#         self.adversarial_criterion = nn.MSELoss()
#         self.l1_criterion = nn.L1Loss()
#         self.checkpoint_dir = '/content/drive/MyDrive/checkpoints'  # Update to Google Drive directory
#         os.makedirs(self.checkpoint_dir, exist_ok=True)

#     def save_checkpoint(self, epoch):
#         torch.save({
#             'epoch': epoch,
#             'generator_state_dict': self.generator.state_dict(),
#             'discriminator_state_dict': self.discriminator.state_dict(),
#             'optimizer_g_state_dict': self.optim_g.state_dict(),
#             'optimizer_d_state_dict': self.optim_d.state_dict(),
#         }, os.path.join(self.checkpoint_dir, f'checkpoint_{epoch}.pth'))

#     def train(self, epochs, batch_size, sample_interval=28, checkpoint_interval=10):
#         valid = torch.ones(batch_size, *self.disc_out).cuda()
#         fake = torch.zeros(batch_size, *self.disc_out).cuda()
#         start_time = datetime.datetime.now()

#         for epoch in range(epochs):
#             for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):
#                 real_A = torch.FloatTensor(imgs_A).cuda()
#                 real_B = torch.FloatTensor(imgs_B).cuda()

#                 # Train Discriminator
#                 self.discriminator.train()
#                 self.optim_d.zero_grad()
#                 fake_A = self.generator(real_B)
#                 fake_A = F.interpolate(fake_A, size=(256, 256))
#                 pred_real = self.discriminator(real_A, real_B)
#                 loss_real = self.adversarial_criterion(pred_real, valid)
#                 pred_fake = self.discriminator(fake_A.detach(), real_B)
#                 loss_fake = self.adversarial_criterion(pred_fake, fake)
#                 d_loss = 0.5 * (loss_real + loss_fake)
#                 d_loss.backward()
#                 self.optim_d.step()

#                 # Train Generator
#                 self.generator.train()
#                 self.optim_g.zero_grad()
#                 pred_fake = self.discriminator(fake_A, real_B)
#                 g_loss_adv = self.adversarial_criterion(pred_fake, valid)
#                 l1_loss = self.l1_criterion(fake_A, real_A)
#                 g_loss = g_loss_adv + 10.0 * l1_loss
#                 g_loss.backward()
#                 self.optim_g.step()

#                 elapsed_time = datetime.datetime.now() - start_time
#                 print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] time: %s"
#                       % (epoch, epochs, batch_i, self.data_loader.n_batches, d_loss.item(), g_loss.item(), elapsed_time))

#                 if batch_i % sample_interval == 0:
#                     self.sample_images(epoch, batch_i, real_A, fake_A)

#             # Save checkpoint
#             if epoch % checkpoint_interval == 0:
#                 self.save_checkpoint(epoch)

#     def sample_images(self, epoch, batch_i, real_A, fake_A):
#         os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
#         r, c = 1, 3  # Adjust grid size to 1x3
#         real_A_np = real_A.permute(0, 2, 3, 1).cpu().data.numpy()
#         fake_A_np = fake_A.permute(0, 2, 3, 1).cpu().data.numpy()
#         fig, axs = plt.subplots(r, c)
#         gen_imgs = np.concatenate([real_A_np, fake_A_np, real_A_np])
#         gen_imgs = 0.5 * gen_imgs + 0.5
#         titles = ['Original', 'Generated', 'Original']
#         cnt = 0
#         for i in range(r):
#             for j in range(c):
#                 axs[j].imshow(gen_imgs[cnt])  # Adjust indexing to axs[j]
#                 axs[j].set_title(titles[j])  # Adjust indexing to titles[j]
#                 axs[j].axis('off')
#                 cnt += 1
#         fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
#         plt.close()

# gan = IDGAN()
# gan.train(epochs=150, batch_size=8, sample_interval=25, checkpoint_interval=10)

# # ...existing code...

######################################################################################################

class IDGAN:
    def __init__(self):
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.img_shape = (self.channels, self.img_rows, self.img_cols)
        self.dataset_name = 'rain'
        self.data_loader = DataLoader(dataset_name=self.dataset_name, img_res=(self.img_rows, self.img_cols))
        self.disc_out = (72, 14, 14)
        self.generator = Generator().to(device)
        self.discriminator = Discriminator().to(device)
        self.optim_g = optim.Adam(self.generator.parameters(), lr=2e-3, betas=(0.9, 0.999))
        self.optim_d = optim.SGD(self.discriminator.parameters(), lr=2e-3, momentum=0.9, weight_decay=1e-6, nesterov=False)
        self.adversarial_criterion = nn.MSELoss()
        self.l1_criterion = nn.L1Loss()
        self.checkpoint_dir = '/content/drive/MyDrive/checkpoints'  # Update to Google Drive directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, epoch):
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optim_g.state_dict(),
            'optimizer_d_state_dict': self.optim_d.state_dict(),
        }, os.path.join(self.checkpoint_dir, f'checkpoint_{epoch}.pth'))

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optim_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optim_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        return checkpoint['epoch']

    def train(self, epochs, batch_size, sample_interval=28, checkpoint_interval=20, resume_from_checkpoint=None):
        start_epoch = 0
        if resume_from_checkpoint:
            start_epoch = self.load_checkpoint(resume_from_checkpoint)
            print(f"Resuming training from epoch {start_epoch}")

        valid = torch.ones(batch_size, *self.disc_out).to(device)
        fake = torch.zeros(batch_size, *self.disc_out).to(device)
        start_time = datetime.datetime.now()

        for epoch in range(start_epoch, epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):
                real_A = torch.FloatTensor(imgs_A).to(device)
                real_B = torch.FloatTensor(imgs_B).to(device)

                # Train Discriminator
                self.discriminator.train()
                self.optim_d.zero_grad()
                fake_A = self.generator(real_B)
                fake_A = F.interpolate(fake_A, size=(256, 256))
                pred_real = self.discriminator(real_A, real_B)
                loss_real = self.adversarial_criterion(pred_real, valid)
                pred_fake = self.discriminator(fake_A.detach(), real_B)
                loss_fake = self.adversarial_criterion(pred_fake, fake)
                d_loss = 0.5 * (loss_real + loss_fake)
                d_loss.backward()
                self.optim_d.step()

                # Train Generator
                self.generator.train()
                self.optim_g.zero_grad()
                pred_fake = self.discriminator(fake_A, real_B)
                g_loss_adv = self.adversarial_criterion(pred_fake, valid)
                l1_loss = self.l1_criterion(fake_A, real_A)
                g_loss = g_loss_adv + 10.0 * l1_loss
                g_loss.backward()
                self.optim_g.step()

                elapsed_time = datetime.datetime.now() - start_time
                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] time: %s" % (epoch, epochs, batch_i, self.data_loader.n_batches, d_loss.item(), g_loss.item(), elapsed_time))

                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i, real_A, fake_A)

            # Save checkpoint
            if epoch % checkpoint_interval == 0:
                self.save_checkpoint(epoch)

    def test(self, checkpoint_path, test_data_loader, output_dir='test_images'):
      # Load the checkpoint
      self.load_checkpoint(checkpoint_path)

      # Create output directory if it doesn't exist
      os.makedirs(output_dir, exist_ok=True)

      # Set the generator to evaluation mode
      self.generator.eval()

      for i, (imgs_A, imgs_B) in enumerate(test_data_loader.load_batch(1)):
          real_A = torch.FloatTensor(imgs_A).to(device)
          real_B = torch.FloatTensor(imgs_B).to(device)

          with torch.no_grad():
              fake_A = self.generator(real_B)
              fake_A = torch.nn.functional.interpolate(fake_A, size=(256, 256))

          self.save_test_images(real_A, fake_A, i, output_dir)

    def save_test_images(self, real_A, fake_A, image_index, output_dir):
        real_A_np = real_A.permute(0, 2, 3, 1).cpu().data.numpy()
        fake_A_np = fake_A.permute(0, 2, 3, 1).cpu().data.numpy()

        gen_imgs = np.concatenate([real_A_np, fake_A_np, real_A_np])
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Generated', 'Original']
        fig, axs = plt.subplots(1, 3)

        for j in range(3):
            axs[j].imshow(gen_imgs[j])
            axs[j].set_title(titles[j])
            axs[j].axis('off')

        fig.savefig(os.path.join(output_dir, f'{image_index}.png'))
        # plt.show()
        plt.close()

    def sample_images(self, epoch, batch_i, real_A, fake_A):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 1, 3  # Adjust grid size to 1x3
        real_A_np = real_A.permute(0, 2, 3, 1).cpu().data.numpy()
        fake_A_np = fake_A.permute(0, 2, 3, 1).cpu().data.numpy()
        fig, axs = plt.subplots(r, c)
        gen_imgs = np.concatenate([real_A_np, fake_A_np, real_A_np])
        gen_imgs = 0.5 * gen_imgs + 0.5
        titles = ['Original', 'Generated', 'Original']
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[j].imshow(gen_imgs[cnt])  # Adjust indexing to axs[j]
                axs[j].set_title(titles[j])  # Adjust indexing to titles[j]
                axs[j].axis('off')
                cnt += 1
        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        # plt.show()
        plt.close()

# gan = IDGAN()

# # To resume training from a checkpoint, provide the path to the checkpoint file
# gan.train(epochs=12500, batch_size=8, sample_interval=25, checkpoint_interval=2, resume_from_checkpoint='/content/drive/MyDrive/checkpoints/checkpoint_434.pth')

# test_data_loader = DataLoader(dataset_name='rain', img_res=(256, 256))
# gan.test(checkpoint_path='/content/drive/MyDrive/checkpoints/checkpoint_434.pth', test_data_loader=test_data_loader)