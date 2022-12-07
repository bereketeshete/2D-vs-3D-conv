
###############################################################
# Import libraries
import sys
import time
sys.path.append("C:/Users/CIRL/AppData/Local\Programs/Python/Python39/Lib/site-packages")
sys.path.append("C:/Users/CIRL/AppData/Local\Programs/Python/Python36/Lib/site-packages")
sys.path.append("C:\ProgramData\Miniconda3\lib\site-packages")

from xlwt import *
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from skimage import io, transform
from unet_model_2D_256 import UNet
from tqdm import tqdm
###############################################################
# Training parameters & paths
TOTAL_EPOCHS = 1000
vol_size = 64
batch_size = 16
learning_rate = 0.001
nors_pfp = 15

train_in_path = "D:/Bereket/DeepLearning - 3D/Data/Data_2D_2/train"
train_gt_path = "D:/Bereket/DeepLearning - 3D/Data/Data_2D_2/train_gt"
valid_in_path = "D:/Bereket/DeepLearning - 3D/Data/Data_2D_2/valid"
valid_gt_path = "D:/Bereket/DeepLearning - 3D/Data/Data_2D_2/valid_gt"
###############################################################
# Output paths
model_output_path = "Generated Models/Generated_Model_2D_2/UNet_SIM%d_2D_actin_epoch_%d_batch_%d_lr_luhong.pkl" %(nors_pfp,TOTAL_EPOCHS,batch_size)
loss_output_path = 'Loss Function/Loss_2D_2/loss_UNet_SIM%d_2D_actin_epoch_%d_batch_%d_lr_luhong.xls' %(nors_pfp,TOTAL_EPOCHS,batch_size)
###############################################################
# Defining classes
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data_in, data_out = sample['image_in'], sample['groundtruth']
        return {'image_in': torch.from_numpy(data_in),
               'groundtruth': torch.from_numpy(data_out)}

class ReconsDataset(torch.utils.data.Dataset):
     def __init__(self, train_in_path,train_gt_path, transform, img_type,in_size):
        self.train_in_path = train_in_path
        self.train_gt_path = train_gt_path
        self.transform = transform
        self.img_type = img_type
        self.in_size = in_size
        self.dirs_gt = os.listdir(self.train_gt_path)
     def __len__(self):
        dirs = os.listdir(self.train_gt_path)   # open the files
        return len(dirs)            # because one of the file is for groundtruth

     def __getitem__(self, idx):
         image_name = os.path.join(self.train_gt_path, self.dirs_gt[idx])
         data_gt = io.imread(image_name)
         #max_out = 15383.0

         data_gt = data_gt/np.max(data_gt)
         
         filepath = os.path.join(self.train_in_path, self.dirs_gt[idx][:-4])
         #filepath = os.listdir(filepath)
         #train_in_size = len(filepath)
         train_in_size = nors_pfp
         
         data_in = np.zeros((train_in_size, vol_size, vol_size))
         filepath = os.path.join(self.train_in_path, self.dirs_gt[idx][:-4])


         if (train_in_size == 15):
             for i in range(train_in_size):
                 image_name = os.path.join(filepath, "HE_"+str(i+1)+"." + self.img_type)
                 image = io.imread(image_name)
                 data_in[i,:,:] = image

         if (train_in_size == 3):
             for i in range(train_in_size):
                 image_name = os.path.join(filepath, "HE_"+str(5*i+1)+"." + self.img_type)
                 image = io.imread(image_name)
                 data_in[i,:,:] = image


         # for i in range(train_in_size):
         #     image_name = os.path.join(filepath, "HE_"+str(i+1)+"." + self.img_type)
         #     image = io.imread(image_name)
         #     data_in[i,:,:] = image



         #max_in = 5315.0
         data_in = data_in/np.max(data_in)
         sample = {'image_in': data_in, 'groundtruth': data_gt}
         
         if self.transform:
             sample = self.transform(sample)
         return sample

def luhong_learning_rate(epoch):
    limits = [3, 8, 12]
    lrs = [1, 0.1, 0.05, 0.005]
    assert len(lrs) == len(limits) + 1
    for lim, lr in zip(limits, lrs):
        if epoch < lim:
            return lr * learning_rate
        return lrs[-1] * learning_rate

def learning_rate_type_1(epoch):

        return  learning_rate

def val_during_training(dataloader):
    model.eval()
    loss_all = np.zeros((len(dataloader)))
    for batch_idx, items in enumerate(dataloader):
        image = items['image_in']
        gt = items['groundtruth']
        #image = np.swapaxes(image, 1,3)
        #image = np.swapaxes(image, 2,3)

        #image = torch.tensor(np.zeros((1, 15, vol_size, vol_size, vol_size)))
        image = image.float()
        image = image.cuda(cuda)

        #gt = torch.tensor(np.zeros((1, 15, vol_size, vol_size, vol_size)))
        gt = gt.squeeze()
        gt = gt.float()
        gt = gt.cuda(cuda)

        pred = model(image).squeeze()
        loss0 =(pred-gt).abs().mean()
        loss_all[batch_idx] = loss0.item()
        
    mae_m, mae_s = loss_all.mean(), loss_all.std()
    return  mae_m, mae_s

if __name__ == "__main__":

    cuda = torch.device('cuda:0')

    # Training data generate from tif to tensor
    train_data = ReconsDataset(train_in_path,
                                train_gt_path,
                                transform = ToTensor(),
                                img_type = 'tif',
                                in_size = vol_size)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)

    # Validation data generate from tif to tensor
    validation_data = ReconsDataset(valid_in_path,
                                valid_gt_path ,
                                transform = ToTensor(),
                                img_type = 'tif',
                                in_size = vol_size)

    validation_dataloader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True, pin_memory=True) # better than for loop


    model = UNet(n_channels=nors_pfp, n_classes=1)
    print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    model.cuda(cuda)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,  betas=(0.9, 0.999))
    loss_all = np.zeros((TOTAL_EPOCHS, 4))
    begin = time.time()


    #############################################################
    ## Training iteration


    for epoch in tqdm(range(TOTAL_EPOCHS), position=0, desc="idx", leave=False, colour='green', ncols=80):

        # mae_mean = mean,
        mean_train, std_train = val_during_training(train_dataloader)
        loss_all[epoch,0] = mean_train
        loss_all[epoch,1] = std_train
        mean_valid, std_valid = val_during_training(validation_dataloader)
        loss_all[epoch,2] = mean_valid
        loss_all[epoch,3] = std_valid


        file = Workbook(encoding = 'utf-8')
        table = file.add_sheet('loss_all')
        for i,p in enumerate(loss_all):
            for j,q in enumerate(p):
                table.write(i,j,q)

        file.save(loss_output_path)

        lr = luhong_learning_rate(epoch)
        #lr = learning_rate
        for p in optimizer.param_groups:
            p['lr'] = lr
            print("learning rate = {}".format(p['lr']))
            
        for batch_idx, items in enumerate(train_dataloader):
            
            image = items['image_in']
            gt = items['groundtruth']

            #image = torch.tensor(np.zeros((1, 15, vol_size, vol_size, vol_size)))
            #gt = torch.tensor(np.zeros((1, 15, vol_size, vol_size, vol_size)))
            model.train()
            image = image.float()
            image = image.cuda(cuda)    
            
            gt = gt.squeeze()
            gt = gt.float()
            gt = gt.cuda(cuda)
            
            pred = model(image).squeeze()

            loss = (pred-gt).abs().mean() + 5 * ((pred-gt)**2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print ("[Epoch %d] [Batch %d/%d] [loss: %f]" % (epoch, batch_idx, len(train_dataloader), loss.item()))

        ##################################################################
        ## Save trained model
        torch.save(model.state_dict(), model_output_path)

        ##################################################################
        ## How long the training took
        time.sleep(1)
        # store end time
        end = time.time()

        # total time taken
        print(f"Total runtime of the program is {(end - begin)/60}")

    with open('Loss Function/Loss_2D_2/loss_UNet_SIM%d_2D_actin_epoch_%d_batch_%d_lr_luhong_log.txt' % (nors_pfp,TOTAL_EPOCHS, batch_size), 'w') as f:
        f.write('Loss Function/Loss_2D_2/loss_UNet_SIM%d_2D_actin_epoch_%d_batch_%d_lr_luhong.xls\n' % (nors_pfp,TOTAL_EPOCHS, batch_size))
        f.write(f"Total runtime of the program is {np.round((end - begin)/60,2)} \n")
        f.write('Adam optimizer, betas=(0.9, 0.999)')
