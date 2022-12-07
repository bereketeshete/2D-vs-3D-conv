
# @author: Bereket Kebede

###############################################################
# Training parameters & paths
TOTAL_EPOCHS = 100
vol_size = 64
batch_size = 1
learning_rate = 0.0001
train_in_size = 3

data_version = 7

train_in_path = "E:/Bereket/Research/DeepLearning - 3D/Data/Data_3D_%d/train"%data_version
train_gt_path = "E:/Bereket/Research/DeepLearning - 3D/Data/Data_3D_%d/train_gt"%data_version
valid_in_path = "E:/Bereket/Research/DeepLearning - 3D/Data/Data_3D_%d/valid"%data_version
valid_gt_path = "E:/Bereket/Research/DeepLearning - 3D/Data/Data_3D_%d/valid_gt"%data_version

min_lr = 3e-5
init_lr = 2e-4

# How many epochs until LR s fully decayed to min_lr
decay_steps = 250

# Learning rate gamma to decay to min_lr after decay_steps
lr_gamma = (min_lr/init_lr) ** (1/(decay_steps))

"""Hyperparameters"""
recon_lam = 1.0
#perp_lam = 0.0005
perp_lam = 0.0