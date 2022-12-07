###############################################################
# Import libraries
import sys
import time
sys.path.append("C:/Users/CIRL/AppData/Local\Programs/Python/Python39/Lib/site-packages")
from xlwt import *
import numpy as np
import os
import torch
from config import *
from torch.utils.data import DataLoader
from torchvision import transforms
from skimage import io
from unet_model_3D import UNet
from tqdm import tqdm
import math
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt


loss_type = 'ryan'
#torch.cuda.empty_cache()

from pretrained_vgg import PretrainedVGG, perceptual_loss

###############################################################
# Defining functions

def psnr(img1, img2, option='single'):
    """
        option = 'single' dataset or 'multiple' dataset
    """
    #img1 = (img1/np.amax(img1))*255
    #img2 = (img2/np.amax(img2))*255
    #mse = torch.mean( (img1 - img2) ** 2, axis=(0,1,2))
    #print('visiting psnr')
    #print(np.shape(img1))
    #print(np.shape(img2))
    if (option == 'single'):
        mse = np.mean( (img1 - img2) ** 2, axis=(0,1,2))
        #print(np.shape(mse))
        if mse.mean == 0:
            return 100
        PIXEL_MAX = 1.0
        final = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    if (option == 'multiple'):
        # do psnr for each
        psnr_all = []
        for i in range(img1.shape[0]):
            mse = np.mean((img1[i] - img2[i]) ** 2, axis=(0, 1, 2))
            if mse.mean == 0:
                return 100
            PIXEL_MAX = 1.0
            temp = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
            psnr_all.append(temp)
        final = np.mean(psnr_all)
    #print(np.shape(mse))
    #print(np.shape(img2))
    # if the images are identical, avoid indeterminate form
    return final

###################################################################
# Estimate Normalized Root Mean Square Error (NRMSE) of two images
def nrmse(img_gt, img2, option = 'single', type="sd"):
    #mse = torch.mean( (img_gt - img2) ** 2 )
    mse = np.mean( (img_gt - img2) ** 2 )
    rmse = math.sqrt(mse)
    if type == "sd":
        #nrmse = rmse/torch.std(img_gt)
        nrmse = rmse/np.std(img_gt)
    if type == "mean":
        nrmse = rmse/np.mean(img_gt)
    if type == "maxmin":
        nrmse = rmse/(np.max(img_gt) - np.min(img_gt))
    if type == "iq":
        nrmse = rmse/ (np.quantile(img_gt, 0.75) - np.quantile(img_gt, 0.25))
    if type not in ["mean", "sd", "maxmin", "iq"]:
        print("Wrong type!")
    return nrmse

###################################################################
# Estimate Structural Similarity Index Measure (SSIM) of two images
def ssim(img1, img2, option='single'):
    # img2 = (img2/img2.max()) * img1.max()
    if (option == 'single'):
        # score = structural_similarity(img1.detach().cpu().numpy(), img2.detach().cpu().numpy())
        score = structural_similarity(img1, img2)
    if (option == 'multiple'):
        score_all = []
        for i in range(img1.shape[0]):
            temp = structural_similarity(img1[i], img2[i])
            score_all.append(temp)  # score = np.mean(score_all)
            #print('ssim: ',score_all[i])
        score = np.mean(score_all)
        #print(len(score_all))
    return score

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data_in, data_out = sample['image_in'], sample['groundtruth']
        return {'image_in': torch.from_numpy(data_in),
               'groundtruth': torch.from_numpy(data_out)}

class CollectDataset(torch.utils.data.Dataset):
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
         #data_gt = data_gt/np.max(data_gt)
         train_in_size = 3
         data_in = np.zeros((train_in_size, 64, 64, 64))
         filepath = os.path.join(self.train_in_path, self.dirs_gt[idx][:-4])


         if (train_in_size == 15):
             for i in range(train_in_size):
                 image_name = os.path.join(filepath, "HE_" + str(i + 1) + "." + self.img_type)
                 image = io.imread(image_name)
                 data_in[i, :, :,:] = image

         if (train_in_size == 3):
             for i in range(train_in_size):
                 image_name = os.path.join(filepath, "HE_" + str(5 * i + 1) + "." + self.img_type)
                 image = io.imread(image_name)
                 data_in[i, :, :,:] = image

         data_in = data_in
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

def eval_during_training(dataloader):
    model.eval()

    performance_all = np.zeros((len(dataloader)))
    #pretrained_vgg = PretrainedVGG().eval()

    #print('length of dataloader',len(dataloader))


    count = 0
    loss_all = np.zeros((len(dataloader)))
    psnr_all = np.zeros(16)
    nrmse_all = np.zeros(16)
    ssim_all = np.zeros(16)
    for batch_idx, items in enumerate(dataloader):
        print('batch_idx : \n',batch_idx)


    for batch_idx, items in enumerate(dataloader):
        #print('batch_idx : \n',batch_idx)
        #if (batch_idx == 0):
        image = items['image_in']
        gt = items['groundtruth']
        print('\n np.shape(image)', np.shape(image))
        print('\n np.shape(gt)', np.shape(gt))

        image = image.float()
        image = image.cuda(cuda)
        gt = gt.float()
        gt = gt.cuda(cuda)
        #gt = gt.squeeze()
        pred = model(image).squeeze()

        print('\n np.shape(pred)', np.shape(pred))
        print('\n np.shape(gt)', np.shape(gt))

        recon_loss0 = (pred - gt).abs().mean() + 5 * ((pred - gt) ** 2).mean()
        loss0 = recon_lam * recon_loss0 #+ perp_lam * perp_loss

        psnr0 = psnr(pred.cpu().detach().numpy(), gt.cpu().detach().numpy())
        nrmse0 = nrmse(pred.cpu().detach().numpy(), gt.cpu().detach().numpy())
        ssim0 = ssim(pred.cpu().detach().numpy(), gt.cpu().detach().numpy())
        psnr_all[count] = psnr0
        nrmse_all[count] = nrmse0
        ssim_all[count] = ssim0
        # psnr_all.append(psnr0)
        # nrmse_all.append(nrmse0)
        # ssim_all.append(ssim0)
        loss_all[batch_idx] = loss0.item()
        count += 1

    #print('\n count', count)



    #print('\n np.shape(pred)', np.shape(pred))
    #print('\n np.shape(gt)', np.shape(gt))

    # for i in range (image.size(dim=1)):
    #     psnr0 = psnr(pred[i].cpu().detach().numpy(), gt[i].cpu().detach().numpy())
    #     nrmse0 = nrmse(pred[i].cpu().detach().numpy(), gt[i].cpu().detach().numpy())
    #     ssim0 = ssim(pred[i].cpu().detach().numpy(), gt[i].cpu().detach().numpy())
    #     psnr_all[i] = psnr0
    #     nrmse_all[i] = nrmse0
    #     ssim_all[i] = ssim0
    #
    #     loss_all[batch_idx] = loss0.item()

        #print('i am inside here')

    #print ('loss_all length', len(loss_all))
    mae_m, mae_s, psnr_m, nrmse_m, ssim_m = loss_all.mean(), loss_all.std(), psnr_all.mean(), nrmse_all.mean(), ssim_all.mean()

    return  mae_m, mae_s, psnr_m, nrmse_m, ssim_m

if __name__ == "__main__":

    cuda = torch.device('cuda:0')

    # Training data generate from tif to tensor
    train_data = CollectDataset(train_in_path,
                                train_gt_path,
                                transform = ToTensor(),
                                img_type = 'tif',
                                in_size = vol_size)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    # Validation data generate from tif to tensor
    validation_data = CollectDataset(valid_in_path,
                                valid_gt_path ,
                                transform = ToTensor(),
                                img_type = 'tif',
                                in_size = vol_size)

    validation_dataloader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=False, pin_memory=True) # better than for loop


    model = UNet(n_channels=train_in_size, n_classes=1)
    #print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    model.cuda(cuda)

    #optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,  betas=(0.9, 0.999))
    optimizer = torch.optim.Adam(model.parameters(),lr=init_lr,  betas=(0.9, 0.999))
    lr_scheduler = StepLR(optimizer, gamma=lr_gamma, step_size=1)  # added from ryan

    #optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate, momentum=0.9)
    loss_all = np.zeros((TOTAL_EPOCHS, 4))
    performance_all = np.zeros((TOTAL_EPOCHS, 6))
    begin = time.time()

    #############################################################
    ## Training 3D U-Net over m iteration
    for epoch in tqdm(range(TOTAL_EPOCHS), position=0, desc="idx", leave=False, colour='green', ncols=80):

        # mae_mean = mean,
        print('evalauating training')
        mean_train, std_train, psnr_train, nrmse_train, ssim_train = eval_during_training(train_dataloader)

        loss_all[epoch,0] = mean_train
        loss_all[epoch,1] = std_train
        performance_all[epoch,0] = psnr_train
        performance_all[epoch,1] = nrmse_train
        performance_all[epoch,2] = ssim_train

        print('evalauating validation')
        mean_valid, std_valid, psnr_valid, nrmse_valid, ssim_valid= eval_during_training(validation_dataloader)
        loss_all[epoch,2] = mean_valid
        loss_all[epoch,3] = std_valid
        performance_all[epoch, 3] = psnr_valid
        performance_all[epoch, 4] = nrmse_valid
        performance_all[epoch, 5] = ssim_valid
        
        file = Workbook(encoding = 'utf-8')
        table = file.add_sheet('loss_all')
        table.write(0, 0, 'LOSS_TRAIN_MEAN')
        table.write(0, 1, 'LOSS_TRAIN_STD')
        table.write(0, 2, 'LOSS_VALID_MEAN')
        table.write(0, 3, 'LOSS_VALID_MEAN')
        for i,p in enumerate(loss_all):
            for j,q in enumerate(p):
                table.write(i+1,j,q)

        file_2 = Workbook(encoding='utf-8')
        table_2 = file_2.add_sheet('performance_all')
        table_2.write(0, 0, 'PSNR_TRAIN')
        table_2.write(0, 1, 'NRMSE_TRAIN')
        table_2.write(0, 2, 'SSIM_TRAIN')
        table_2.write(0, 3, 'PSNR_VALID')
        table_2.write(0, 4, 'NRMSE_VALID')
        table_2.write(0, 5, 'SSIM_VALID')
        for i,p in enumerate(performance_all):
            for j,q in enumerate(p):
                table_2.write(i+1,j,q)

        #file.save('Loss Function/Loss_3D_3/20210420_H9C2-dTag_GLU_37C_1520_sim-fast_005/loss_UNet_SIM3_20210420_H9C2-dTag_GLU_37C_1520_sim-fast_005_epoch_%d_batch_%d_lr_luhong_const_lr_%.6f.xls' %(TOTAL_EPOCHS,batch_size, learning_rate))

        loss_path = 'E:/Bereket/Research/DeepLearning - 3D/Training_codes/UNet/Loss function/Loss_3D_%d/'%data_version
        performance_path = 'E:/Bereket/Research/DeepLearning - 3D/Training_codes/UNet/Performance function/Performance_3D_%d/'%data_version

        if (os.path.isdir(loss_path) == False):
            os.makedirs(loss_path)
        if (os.path.isdir(performance_path) == False):
            os.makedirs(performance_path)

        file.save(loss_path + 'loss_UNet_SIM3_Data_3D_%d_epoch_%d_batch_%d_lr_ryan_lr_%.4f.xls' %(data_version,TOTAL_EPOCHS, batch_size, learning_rate))
        file_2.save(performance_path + 'performance_UNet_SIM3_Data_3D_%d_epoch_%d_batch_%d_lr_ryan_lr_%.4f.xls' % (data_version, TOTAL_EPOCHS, batch_size, learning_rate))

        #lr = luhong_learning_rate(epoch)
        #lr = learning_rate

        for p in optimizer.param_groups:
            #p['lr'] = lr
            cur_lr = p['lr']
            #print("learning rate = {}".format(p['lr']))
            #print("learning rate = {}".format(cur_lr))

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

            #psnr_value = psnr(pred[0].cpu().detach().numpy(),gt.cpu().detach().numpy())
            #ssim_value = nrmse(pred[0].cpu().detach().numpy(),gt.cpu().detach().numpy())
            #nrmse_value = nrmse(pred[0].cpu().detach().numpy(), gt.cpu().detach().numpy())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        lr_scheduler.step()  # added from ryan
            #print ("[Epoch %d] [Batch %d/%d] [loss: %f]" % (epoch, batch_idx, len(train_dataloader), loss.item()))

        ##################################################################
        ## Save trained model
        #torch.save(model.state_dict(), "Generated Models/Generated_Model_3D_3/20210420_H9C2-dTag_GLU_37C_1520_sim-fast_005/UNet_SIM3_3D_20210420_H9C2-dTag_GLU_37C_1520_sim-fast_005_epoch_%d_batch_%d_luhong_const_lr_%.6f.pkl" %(TOTAL_EPOCHS,batch_size, learning_rate))
        model_output_path = "E:/Bereket/Research/DeepLearning - 3D/Training_codes/UNet/Generated Models/Generated_Model_3D_%d"%data_version

        if (os.path.isdir(model_output_path) == False):
            os.makedirs(model_output_path)
        #torch.save(model.state_dict(), model_output_path+"/20210420_H9C2-dTag_GLU_37C_1520_sim-fast_005/UNet_SIM3_3D_20210420_H9C2-dTag_GLU_37C_1520_sim-fast_005_epoch_%d_batch_%d_const_lr_%.4f.pkl" %(TOTAL_EPOCHS,batch_size, learning_rate))
        torch.save(model.state_dict(), model_output_path+"UNet_SIM3_3D_Data_3D_%d_epoch_%d_batch_%d_lr_ryan_%.4f.pkl" %(data_version,TOTAL_EPOCHS,batch_size, learning_rate))

        ##################################################################
        ## How long the training took
        time.sleep(1)
        # store end time
        end = time.time()

        # total time taken
        #print(f"Total runtime of the program is {round((end - begin)/60,2)} min")

    log_path = 'E:/Bereket/Research/DeepLearning - 3D/Training_codes/UNet/Generated Models/Generated_Model_3D_%d/20210420_H9C2-dTag_GLU_37C_1520_sim-fast_005/logs'%data_version

    if (os.path.isdir(log_path) == False):
        os.makedirs(log_path)

    with open(log_path+'/UNet_%d_Data_3D_%d_%d_batch_%d_lr_ryan_%.4f_log.txt' % (train_in_size, data_version,TOTAL_EPOCHS, batch_size, learning_rate), 'w') as f:

        f.write('E:/Bereket/Research/DeepLearning - 3D/Training_codes/UNet/Generated Models/Generated_Model_3D_12/20210420_H9C2-dTag_GLU_37C_1520_sim-fast_005/UNet_SIM3_3D_20210420_H9C2-dTag_GLU_37C_1520_sim-fast_005_epoch_%d_batch_%d_lr_ryan_%.4f.pkl \n' %(TOTAL_EPOCHS,batch_size, learning_rate))
        f.write('E:/Bereket/Research/DeepLearning - 3D/Training_codes/UNet/Loss function/Loss_3D_12/20210420_H9C2-dTag_GLU_37C_1520_sim-fast_005/loss_UNet_SIM3_20210420_H9C2-dTag_GLU_37C_1520_sim-fast_005_epoch_%d_batch_%d_lr_ryan_%.4f.xls \n' %(TOTAL_EPOCHS,batch_size, learning_rate))
        f.write('Data used for training: top half of of Data_3D_12 \n')
        f.write(f"Total runtime of the program is {np.round((end - begin)/60,2)} \n")
        f.write('Adam optimizer, betas=(0.9, 0.999) \n')
        f.write('%d RawSIM channels' %train_in_size)