import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1

import torch
from PIL import Image
from torchvision import transforms
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
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

class IDGAN:
    def __init__(self):
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.img_shape = (self.channels, self.img_rows, self.img_cols)
        self.dataset_name = 'rain'
        self.data_loader = DataLoader(dataset_name=self.dataset_name, img_res=(self.img_rows, self.img_cols))
        self.disc_out = (72, 14, 14)
        self.generator = Generator().cuda()
        self.discriminator = Discriminator().cuda()
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

        valid = torch.ones(batch_size, *self.disc_out).cuda()
        fake = torch.zeros(batch_size, *self.disc_out).cuda()
        start_time = datetime.datetime.now()

        for epoch in range(start_epoch, epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):
                real_A = torch.FloatTensor(imgs_A).cuda()
                real_B = torch.FloatTensor(imgs_B).cuda()

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
          real_A = torch.FloatTensor(imgs_A).cuda()
          real_B = torch.FloatTensor(imgs_B).cuda()

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

class MixedImgTripletDataset(Dataset):
    def __init__(self, dataset_folder, transform=None, gan_checkpoint_path=None):
        self.clean_images_folder = os.path.join(dataset_folder, 'clean')
        self.rainy_images_folder = os.path.join(dataset_folder, 'rainy')
        self.transform = transform

        # Initialize GAN
        if gan_checkpoint_path:
            self.gan = Generator().cuda()
            checkpoint = torch.load(gan_checkpoint_path)
            self.gan.load_state_dict(checkpoint['generator_state_dict'])
            self.gan.eval()
        else:
            self.gan = None

    def process_with_gan(self, img):
        # Convert PIL image to tensor
        if not isinstance(img, torch.Tensor):
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            img = transform(img)

        # Process through GAN
        with torch.no_grad():
            img = img.unsqueeze(0).cuda()
            processed = self.gan(img)
            processed = F.interpolate(processed, size=(256, 256))
            processed = processed.squeeze(0).cpu()

        # Convert back to PIL image for further transformations
        processed = processed * 0.5 + 0.5  # denormalize
        processed = processed.clamp(0, 1)
        processed = transforms.ToPILImage()(processed)
        return processed

    def load_img(self, path):
        img = Image.open(path).convert('RGB')

        # Process rainy images through GAN
        if self.gan and 'rainy' in path:
            img = self.process_with_gan(img)

        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(os.listdir(self.clean_images_folder))

    # def load_img(self, path):
    #     img = Image.open(path).convert('RGB')
    #     if self.transform:
    #         img = self.transform(img)
    #     return img

    def __getitem__(self, idx):
        folder_name = os.listdir(self.clean_images_folder)[idx]

        # Anchor: Randomly from clean or rainy dataset
        if np.random.rand() > 0.5:
            img1_folder_path = os.path.join(self.clean_images_folder, folder_name)
        else:
            img1_folder_path = os.path.join(self.rainy_images_folder, folder_name)
        img1_image = os.listdir(img1_folder_path)
        img1_idx = np.random.randint(0, len(img1_image))
        anchor_path = os.path.join(img1_folder_path, img1_image[img1_idx])

        # Positive: Randomly from clean or rainy dataset
        if np.random.rand() > 0.5:
            img2_folder_path = os.path.join(self.clean_images_folder, folder_name)
        else:
            img2_folder_path = os.path.join(self.rainy_images_folder, folder_name)
        img2_image = os.listdir(img2_folder_path)
        positive_idx = np.random.randint(0, len(img2_image))
        positive_path = os.path.join(img2_folder_path, img2_image[positive_idx])

        # Negative: Randomly from clean or rainy dataset, different identity
        if np.random.rand() > 0.5:
            negative_folder_root = self.clean_images_folder
        else:
            negative_folder_root = self.rainy_images_folder
        negative_folder_name = np.random.choice(
            [folder for folder in os.listdir(negative_folder_root) if folder != folder_name]
        )
        img3_folder_path = os.path.join(negative_folder_root, negative_folder_name)
        img3_image = os.listdir(img3_folder_path)
        negative_idx = np.random.randint(0, len(img3_image))
        negative_path = os.path.join(img3_folder_path, img3_image[negative_idx])

        # Load images
        anchor = self.load_img(anchor_path)
        positive = self.load_img(positive_path)
        negative = self.load_img(negative_path)

        return anchor, positive, negative

class TripletNetwork(nn.Module):
    def __init__(self, device):
        super(TripletNetwork, self).__init__()
        self.resnet = InceptionResnetV1(pretrained='vggface2').to(device)

    def forward_once(self, x):
        return self.resnet(x)

    def forward(self, anchor, positive, negative):
        anchor_output = self.forward_once(anchor)
        positive_output = self.forward_once(positive)
        negative_output = self.forward_once(negative)
        return anchor_output, positive_output, negative_output

class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        loss = torch.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()



def evaluate_triplet_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_triplets = 0
    total_triplets = 0

    with torch.no_grad():
        for anchor, positive, negative in tqdm(data_loader, desc="Evaluating"):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            anchor_out, positive_out, negative_out = model(anchor, positive, negative)

            loss = criterion(anchor_out, positive_out, negative_out)
            total_loss += loss.item()

            pos_dist = F.pairwise_distance(anchor_out, positive_out, p=2)
            neg_dist = F.pairwise_distance(anchor_out, negative_out, p=2)
            correct_triplets += torch.sum(pos_dist < neg_dist).item()
            total_triplets += anchor.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_triplets / total_triplets
    return avg_loss, accuracy



def train_with_triplet_loss_and_accuracy(
    model, train_loader, test_loader, criterion, optimizer, num_epochs, device, checkpoint_path=None
):
    if checkpoint_path:
        try:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Loaded checkpoint from epoch {start_epoch}")
        except FileNotFoundError:
            print("No checkpoint found. Starting training from scratch.")
            start_epoch = 1
    else:
        start_epoch = 1

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        total_loss = 0
        correct_triplets = 0
        total_triplets = 0

        for anchor, positive, negative in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            # Forward pass
            anchor_out, positive_out, negative_out = model(anchor, positive, negative)

            # Compute loss
            loss = criterion(anchor_out, positive_out, negative_out)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pos_dist = F.pairwise_distance(anchor_out, positive_out, p=2)
            neg_dist = F.pairwise_distance(anchor_out, negative_out, p=2)
            correct_triplets += torch.sum(pos_dist < neg_dist).item()
            total_triplets += anchor.size(0)

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = correct_triplets / total_triplets
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")

        val_loss, val_accuracy = evaluate_triplet_model(model, test_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")


        if checkpoint_path is not None:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved as {checkpoint_path}")

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transforms
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


# Initialize the face recognition model
model = TripletNetwork(device).to(device)
criterion = TripletLoss(margin=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# num_epochs = 10
# checkpoint_path = "triplet_loss_checkpoint_mixed.pth"

# train_with_triplet_loss_and_accuracy(
#     model, train_loader, test_loader, criterion, optimizer, num_epochs, device, checkpoint_path
# )


# Train the model
num_epochs = 10
checkpoint_path = "/content/drive/MyDrive/checkpoints/triplet_loss_checkpoint_mixed_with_gan.pth"

train_with_triplet_loss_and_accuracy(
    model, train_loader, test_loader, criterion, optimizer, num_epochs, device, checkpoint_path
)

import torch

# Assume your model is already trained and available as `model`
# Move to CPU before saving (optional but good practice for portability)
model_cpu = model.to('cpu')

# Save the entire model (architecture + weights)
torch.save(model_cpu, '/content/drive/MyDrive/checkpoints/triplet_model_full.pt')

print("✅ Model saved as 'triplet_model_full.pt' successfully!")

