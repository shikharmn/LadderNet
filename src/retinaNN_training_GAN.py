###################################################
#
#   Script to:
#   - Load the images and extract the patches
#   - Define the neural network
#   - define the training
#
##################################################


import numpy as np
from six.moves import configparser
import torch.backends.cudnn as cudnn

import sys
sys.path.insert(0, '../')
from lib.help_functions import *

#function to obtain data for training/testing (validation)
from lib.extract_patches import get_data_training
import os

from losses import *
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
import random
from sklearn.metrics import f1_score

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#=========  Load settings from Config file
config = configparser.RawConfigParser()
config.read('../configuration.txt')

#patch to the datasets
path_data = config.get('data paths', 'path_local')
#Experiment name
name_experiment = config.get('experiment name', 'name')
#training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))

#========== Define parameters here =============================
# log file
if not os.path.exists('./logs'):
    os.mkdir('logs')

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

start_epoch = 0  # start from epoch 0 or last checkpoint epoch
total_epoch = 300
val_portion = 0.1

lr_epoch = np.array([150,total_epoch])
lr_value= np.array([0.001,0.0001])

layers = 4
filters = 10

from LadderNetv65 import LadderNetv6
from Discriminator import Discriminator
net = LadderNetv6(num_classes=2,layers=layers,filters=filters,inplanes=1)
D = Discriminator(planes=filters,layers=layers,inplanes=1).to(device)

wandb.watch(net)
wandb.watch(D)

print("Toral number of parameters: "+str(count_parameters(net)))

check_path = 'LadderNetv65_layer_%d_filter_%d.pt7'% (layers,filters) #'UNet16.pt7'#'UNet_Resnet101.pt7'
resume = False
criterion = LossMulti(jaccard_weight=0)
criterion_D = nn.BCEWithLogitsLoss()
#criterion = CrossEntropy2d()

#optimizer = optim.SGD(net.parameters(), momentum=0.9,
#                      lr=0.01, weight_decay=5e-4, nesterov=True)
optimizer_G = optim.Adam(net.parameters(),lr=0.005)
optimizer_D = optim.Adam(D.parameters(),lr=0.005)

#============ Load the data and divided in patches

patches_imgs_train, patches_masks_train = get_data_training(
    DRIVE_train_imgs_original = path_data + config.get('data paths', 'train_imgs_original'),
    DRIVE_train_groudTruth = path_data + config.get('data paths', 'train_groundTruth'),  #masks
    patch_height = int(config.get('data attributes', 'patch_height')),
    patch_width = int(config.get('data attributes', 'patch_width')),
    N_subimgs = int(config.get('training settings', 'N_subimgs')),
    inside_FOV = config.getboolean('training settings', 'inside_FOV') #select the patches only inside the FOV  (default == True)
)

val_ind = random.sample(range(patches_masks_train.shape[0]),int(np.floor(val_portion*patches_masks_train.shape[0])))
train_ind =  set(range(patches_masks_train.shape[0])) - set(val_ind)
train_ind = list(train_ind)

#Save the dataset to drive to avoid RAM overload and clear RAM whenever possible
np.save('train_imgs', patches_imgs_train[train_ind,...])
np.save('val_imgs', patches_imgs_train[val_ind,...])
patches_imgs_train = None

np.save('train_masks', patches_masks_train[train_ind,...])
np.save('val_masks', patches_masks_train[val_ind,...])
patches_masks_train = None



class TrainDataset(Dataset):
    """Endovis 2018 dataset."""

    def __init__(self, mode):

        self.imgs = np.load(mode + '_imgs.npy')
        self.masks = np.load(mode+'_masks.npy')

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        tmp = self.masks[idx]
        tmp = np.squeeze(tmp,0)
        return torch.from_numpy(self.imgs[idx,...]).float(), torch.from_numpy(tmp).long()

# Create the dataloaders for training and validation

train_set = TrainDataset('train')
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

val_set = TrainDataset('val')
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

#========= Save a sample of what you're feeding to the neural network ==========
N_sample = 40

best_loss = np.Inf

# create a list of learning rate with epochs
#========= Save a sample of what you're feeding to the neural network ==========
N_sample = 40
best_loss = np.Inf

# create a list of learning rate with epochs
lr_schedule = np.zeros(total_epoch)
for l in range(len(lr_epoch)):
    if l ==0:
        lr_schedule[0:lr_epoch[l]] = lr_value[l]
    else:
        lr_schedule[lr_epoch[l-1]:lr_epoch[l]] = lr_value[l]

if device != 'cpu':
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    D = torch.nn.DataParallel(D, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

if resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+check_path)
    net.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']

# custom weights initialization called on net and D
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

net = net.apply(weights_init)
D = D.apply(weights_init)

g_steps = 0
d_steps = 0

def train_G_naive(epoch):
    global steps

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    IoU = []

    # get learning rate from learing schedule
    lr = lr_schedule[epoch]
    for param_group in optimizer_G.param_groups:
        param_group['lr'] = lr

    print("Learning rate = %4f\n" % optimizer_G.param_groups[0]['lr'])

    IU = []
    # train network
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer_G.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer_G.step()

        train_loss += loss.item()
        wandb.log({'train-loss_G': loss.item(), 'steps': steps, 'epoch': epoch})
        steps += 1

    print("Epoch %d: Train loss %4f\n" % (epoch, train_loss / np.float32(len(train_loader))))

def train_G(epoch):

    sigmoid = nn.Sigmoid()
    global g_steps
    print('\nEpoch: %d' % epoch)
    net.train()
    D.eval()
    train_loss = 0
    IoU = []

    # get learning rate from learing schedule
    lr = lr_schedule[epoch]
    for param_group in optimizer_G.param_groups:
        param_group['lr'] = lr

    print("Learning rate = %4f\n" % optimizer_G.param_groups[0]['lr'])
    label_real = torch.full((batch_size,), 1, dtype=torch.float, device=device)

    IU = []
    D_G_z = 0.0
    # train network
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer_G.zero_grad()

        outputs = net(inputs)
        M_inputs = torch.exp(outputs[:,1]) * targets
        D_preds = D(M_inputs.unsqueeze(1))
        loss = criterion(outputs, targets) + criterion_D(D_preds.reshape(batch_size), label_real)
        loss.backward()
        optimizer_G.step()
        D_G_z_step = torch.mean(sigmoid(D_preds))
        D_G_z += D_G_z_step

        train_loss += loss.item()
        wandb.log({'train-loss_G': loss.item(), 'D(G(z))': D_G_z_step,
                   'gen_steps': g_steps, 'epoch': epoch})
        g_steps += 1

    print("Epoch %d: Train loss %4f and D(G(z)) is %4f\n" % (epoch, train_loss / np.float32(len(train_loader)), D_G_z / np.float32(len(train_loader))))

def train_D(epoch):

    sigmoid = nn.Sigmoid()
    global d_steps

    print('\nEpoch: %d' % epoch)
    D.train()
    net.eval()
    train_loss = 0
    IoU = []

    # get learning rate from learing schedule
    lr = lr_schedule[epoch]
    for param_group in optimizer_D.param_groups:
        param_group['lr'] = lr

    print("Learning rate = %4f\n" % optimizer_D.param_groups[0]['lr'])
    label_real = torch.full((batch_size,), 1, dtype=torch.float, device=device)
    label_fake = torch.full((batch_size,), 0, dtype=torch.float, device=device)

    D_x = 0.0

    IU = []
    # train network
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer_D.zero_grad()

        outputs = net(inputs)
        MI_real = inputs[:,0] * targets
        MI_fake = outputs[:,1] * inputs[:,0]

        out_real = D(MI_real.unsqueeze(1))
        out_fake = D(MI_fake.unsqueeze(1))
        loss_fake = criterion_D(out_fake.reshape(batch_size), label_fake)
        loss_fake.backward()
        loss_real = criterion_D(out_real.reshape(batch_size), label_real)
        loss_real.backward()
        optimizer_D.step()

        D_x_step = torch.mean(sigmoid(out_real))
        D_x += D_x_step

        train_loss += loss_real.item() + loss_fake.item()

        wandb.log({'train-loss_D': loss_real.item()+loss_fake.item(), 'D(x)': D_x_step,
                   'disc_steps': d_steps, 'epoch': epoch})
        d_steps += 1

    print("Epoch %d: Train loss %4f and D(x) is %4f\n" % (epoch, train_loss / np.float32(len(train_loader)), D_x / np.float32(len(train_loader))))


def test(epoch, display=False):
    global best_loss
    global steps
    net.eval()
    test_loss = 0
    with torch.no_grad():

        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            wandb.log({'val-loss_D': loss.item(), 'epoch': epoch})
            steps += 1
        print(
            'Valid loss: {:.4f}'.format(test_loss))

    # Save checkpoint.
    if test_loss < best_loss:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'best_loss': best_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + check_path)
        best_loss = test_loss

for epoch in range(0,100):
    train_G_naive(epoch)
    test(epoch)

for epoch in range(100,200):
    train_D(epoch)
    train_G(epoch)
    test(epoch)