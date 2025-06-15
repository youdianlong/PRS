from locale import normalize
from multiprocessing import reduction
import pdb
from turtle import pd
import numpy as np
import torch.nn as nn
import torch
import random
from utils.losses import mask_DiceLoss
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

DICE = mask_DiceLoss(nclass=2)
CE = nn.CrossEntropyLoss(reduction='none')

def obtain_cutmix_box_3d(img, p=0.5):
    """
    在三维图像中随机生成一个立方体区域，并将该区域的掩码设置为0。

    参数:
        img (torch.Tensor): 输入图像张量，形状为 (batch_size, channel, img_x, img_y, img_z)。
        p (float): 执行CutMix的概率。

    返回:
        mask (torch.Tensor): 生成的掩码张量，形状为 (img_x, img_y, img_z)。
        loss_mask (torch.Tensor): 用于损失计算的掩码张量，形状为 (batch_size, img_x, img_y, img_z)。
    """
    batch_size, channel, img_x, img_y, img_z = img.size()

    # 初始化掩码为全1
    mask = torch.ones(img_x, img_y, img_z).cuda()
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()

    # 定义随机立方体大小的范围
    size_min = 0.02
    size_max = 0.4
    ratio_min = 0.3
    ratio_max = 1 / 0.3

    if random.random() > p:
        return mask.long(), loss_mask.long()

    # 随机确定立方体的总体积
    size = np.random.uniform(size_min, size_max) * img_x * img_y * img_z

    # 尝试找到一个合适的立方体比例和尺寸
    for _ in range(10):
        ratio_xy = np.random.uniform(ratio_min, ratio_max)
        ratio_xz = np.random.uniform(ratio_min, ratio_max)
        ratio_yz = np.random.uniform(ratio_min, ratio_max)

        # 计算各个维度的长度
        cutmix_w = int(round((size / (ratio_xy * ratio_xz * ratio_yz)) ** (1/3) * ratio_xy))
        cutmix_h = int(round((size / (ratio_xy * ratio_xz * ratio_yz)) ** (1/3) * ratio_xz))
        cutmix_d = int(round((size / (ratio_xy * ratio_xz * ratio_yz)) ** (1/3) * ratio_yz))

        # 确保尺寸不超过图像维度
        if cutmix_w < img_x and cutmix_h < img_y and cutmix_d < img_z:
            break
    else:
        # 如果尝试多次仍未找到合适的尺寸，则使用最小尺寸
        cutmix_w = int(round((size / 3) ** (1/3)))
        cutmix_h = cutmix_w
        cutmix_d = cutmix_w

    # 随机选择立方体的起始位置，确保不超出边界
    x = np.random.randint(0, max(img_x - cutmix_w, 1))
    y = np.random.randint(0, max(img_y - cutmix_h, 1))
    z = np.random.randint(0, max(img_z - cutmix_d, 1))

    # 设置掩码
    mask[x:x + cutmix_w, y:y + cutmix_h, z:z + cutmix_d] = 0
    loss_mask[:, x:x + cutmix_w, y:y + cutmix_h, z:z + cutmix_d] = 0

    return mask.long(), loss_mask.long()

def context_mask(img, mask_ratio):
    batch_size, channel, img_x, img_y, img_z = img.shape[0],img.shape[1],img.shape[2],img.shape[3],img.shape[4]
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()
    patch_pixel_x, patch_pixel_y, patch_pixel_z = int(img_x*mask_ratio), int(img_y*mask_ratio), int(img_z*mask_ratio)
    w = np.random.randint(0, 112 - patch_pixel_x)
    h = np.random.randint(0, 112 - patch_pixel_y)
    z = np.random.randint(0, 80 - patch_pixel_z)
    mask[w:w+patch_pixel_x, h:h+patch_pixel_y, z:z+patch_pixel_z] = 0
    loss_mask[:, w:w+patch_pixel_x, h:h+patch_pixel_y, z:z+patch_pixel_z] = 0
    return mask.long(), loss_mask.long()

def random_mask(img):
    batch_size, channel, img_x, img_y, img_z = img.shape[0],img.shape[1],img.shape[2],img.shape[3],img.shape[4]
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()
    patch_pixel_x, patch_pixel_y, patch_pixel_z = int(img_x*2/3), int(img_y*2/3), int(img_z*2/3)
    mask_num = 27
    mask_size_x, mask_size_y, mask_size_z = int(patch_pixel_x/3)+1, int(patch_pixel_y/3)+1, int(patch_pixel_z/3)
    size_x, size_y, size_z = int(img_x/3), int(img_y/3), int(img_z/3)
    for xs in range(3):
        for ys in range(3):
            for zs in range(3):
                w = np.random.randint(xs*size_x, (xs+1)*size_x - mask_size_x - 1)
                h = np.random.randint(ys*size_y, (ys+1)*size_y - mask_size_y - 1)
                z = np.random.randint(zs*size_z, (zs+1)*size_z - mask_size_z - 1)
                mask[w:w+mask_size_x, h:h+mask_size_y, z:z+mask_size_z] = 0
                loss_mask[:, w:w+mask_size_x, h:h+mask_size_y, z:z+mask_size_z] = 0
    return mask.long(), loss_mask.long()

def concate_mask(img):
    batch_size, channel, img_x, img_y, img_z = img.shape[0],img.shape[1],img.shape[2],img.shape[3],img.shape[4]
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()
    z_length = int(img_z * 8 / 27)
    z = np.random.randint(0, img_z - z_length -1)
    mask[:, :, z:z+z_length] = 0
    loss_mask[:, :, :, z:z+z_length] = 0
    return mask.long(), loss_mask.long()

def mix_loss(net3_output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    dice_loss = DICE(net3_output, img_l, mask) * image_weight 
    dice_loss += DICE(net3_output, patch_l, patch_mask) * patch_weight
    loss_ce = image_weight * (CE(net3_output, img_l) * mask).sum() / (mask.sum() + 1e-16) 
    loss_ce += patch_weight * (CE(net3_output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)
    loss = (dice_loss + loss_ce) / 2
    return loss

def sup_loss(output, label):
    label = label.type(torch.int64)
    dice_loss = DICE(output, label)
    loss_ce = torch.mean(CE(output, label))
    loss = (dice_loss + loss_ce) / 2
    return loss

@torch.no_grad()
def update_ema_variables(model, ema_model, alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)

@torch.no_grad()
def update_ema_students(model1, model2, ema_model, alpha):
    for ema_param, param1, param2 in zip(ema_model.parameters(), model1.parameters(), model2.parameters()):
        ema_param.data.mul_(alpha).add_(((1 - alpha)/2) * param1.data).add_(((1 - alpha)/2) * param2.data)

@torch.no_grad()
def parameter_sharing(model, ema_model):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data = param.data

class BBoxException(Exception):
    pass

def get_non_empty_min_max_idx_along_axis(mask, axis):
    """
    Get non zero min and max index along given axis.
    :param mask:
    :param axis:
    :return:
    """
    if isinstance(mask, torch.Tensor):
        # pytorch is the axis you want to get
        nonzero_idx = (mask != 0).nonzero()
        if len(nonzero_idx) == 0:
            min = max = 0
        else:
            max = nonzero_idx[:, axis].max()
            min = nonzero_idx[:, axis].min()
    elif isinstance(mask, np.ndarray):
        nonzero_idx = (mask != 0).nonzero()
        if len(nonzero_idx[axis]) == 0:
            min = max = 0
        else:
            max = nonzero_idx[axis].max()
            min = nonzero_idx[axis].min()
    else:
        raise BBoxException("Wrong type")
    max += 1
    return min, max


def get_bbox_3d(mask):
    """ Input : [D, H, W] , output : ((min_x, max_x), (min_y, max_y), (min_z, max_z))
    Return non zero value's min and max index for a mask
    If no value exists, an array of all zero returns
    :param mask:  numpy of [D, H, W]
    :return:
    """
    assert len(mask.shape) == 3
    min_z, max_z = get_non_empty_min_max_idx_along_axis(mask, 2)
    min_y, max_y = get_non_empty_min_max_idx_along_axis(mask, 1)
    min_x, max_x = get_non_empty_min_max_idx_along_axis(mask, 0)

    return np.array(((min_x, max_x),
                     (min_y, max_y),
                     (min_z, max_z)))

def get_bbox_mask(mask):
    batch_szie, x_dim, y_dim, z_dim = mask.shape[0], mask.shape[1], mask.shape[2], mask.shape[3]
    mix_mask = torch.ones(batch_szie, 1, x_dim, y_dim, z_dim).cuda()
    for i in range(batch_szie):
        curr_mask = mask[i, ...].squeeze()
        (min_x, max_x), (min_y, max_y), (min_z, max_z) = get_bbox_3d(curr_mask)
        mix_mask[i, :, min_x:max_x, min_y:max_y, min_z:max_z] = 0
    return mix_mask.long()

class meanIOU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def add_batch(self, pred, gt):
        pred = pred.cpu().numpy().flatten()
        gt = gt.cpu().numpy().flatten()
        mask = (gt >= 0) & (gt < self.num_classes)
        hist = np.bincount(self.num_classes * gt[mask].astype(int) + pred[mask], minlength=self.num_classes ** 2)
        hist = hist.reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += hist

    def evaluate(self):
        intersection = np.diag(self.confusion_matrix)
        ground_truth_set = self.confusion_matrix.sum(axis=1)
        predicted_set = self.confusion_matrix.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection
        iu = intersection / (union + 1e-16)
        return iu, self.confusion_matrix

