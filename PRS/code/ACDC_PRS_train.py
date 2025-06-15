import argparse
from asyncore import write
from decimal import ConversionSyntax
import logging
from multiprocessing import reduction
import os
import random
import shutil
import sys
import time
import pdb
import cv2
import matplotlib.pyplot as plt
import imageio

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.functional import embedding
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
from tqdm import tqdm
from skimage.measure import label
from skimage.exposure import match_histograms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler, ThreeStreamBatchSampler,WeakStrongGenerator)
from networks.net_factory import PRS_net, net_factory
from utils import losses, ramps, feature_memory, contrastive_losses, val_2d
from utils.PRS_utils import meanIOU

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data2/zxm/Projects/PRS/data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='PRS_hym', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--pre_iterations', type=int, default=10000, help='maximum epoch number to train')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--image_size', type=list,  default=[256, 256], help='image size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4, help='output channel of network')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12, help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=7, help='labeled data')
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
# costs
parser.add_argument('--gpu', type=str,  default='2', help='GPU to use')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float,  default='6.0', help='magnitude')
parser.add_argument('--s_param', type=int,  default=6, help='multinum of random masks')
parser.add_argument("--cutmix_prob", default=1, type=float)
# patch size
parser.add_argument('--patch_size', type=int, default=64, help='patch_size')
parser.add_argument( '--mask_density', type=float, default=0.4, help='mask_density')
parser.add_argument( '--preserve_fraction', type=float, default=0.2, help='preserve_fraction')

args = parser.parse_args()

dice_loss = losses.DiceLoss(n_classes=4)

def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def save_net_opt(net, optimizer, path):
    state = {
        'net':net.state_dict(),
        'opt':optimizer.state_dict(),
    }
    torch.save(state, str(path))

def get_ACDC_LargestCC(segmentation):
    class_list = []
    for i in range(1, 4):
        temp_prob = segmentation == i * torch.ones_like(segmentation)
        temp_prob = temp_prob.detach().cpu().numpy()
        labels = label(temp_prob)
        # -- with 'try'
        assert(labels.max() != 0)  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        class_list.append(largestCC * i)
    acdc_largestCC = class_list[0] + class_list[1] + class_list[2]
    return torch.from_numpy(acdc_largestCC).cuda()

def get_ACDC_2DLargestCC(segmentation):
    batch_list = []
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, 4):
            temp_seg = segmentation[i] #== c *  torch.ones_like(segmentation[i])
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)          
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)
        
        n_batch = class_list[0] + class_list[1] + class_list[2]
        batch_list.append(n_batch)

    return torch.Tensor(batch_list).cuda()


def get_ACDC_masks(output, nms=0):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    if nms == 1:
        probs = get_ACDC_2DLargestCC(probs)      
    return probs

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 5* args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)

def histogram_match_batch(img_a, img_b):
    """
    对批量图像进行直方图匹配，将 img_a 的直方图匹配到 img_b。

    参数:
    - img_a: 张量，形状为 (batch_size, channels, H, W)
    - img_b: 张量，形状为 (batch_size, channels, H, W)

    返回:
    - matched_a: 张量，形状为 (batch_size, channels, H, W)
    """
    img_a_np = img_a.cpu().numpy()
    img_b_np = img_b.cpu().numpy()
    matched_a = np.empty_like(img_a_np)
    for i in range(img_a_np.shape[0]):
        for c in range(img_a_np.shape[1]):
            matched_a[i, c] = match_histograms(img_a_np[i, c], img_b_np[i, c], channel_axis=-1)
    return torch.from_numpy(matched_a).cuda()

def visualize_tsne(model, data_loader, writer, iter_num, device='cuda', max_points=2000):
    model.eval()  # 暂时切换到 eval 模式

    all_features = []
    all_labels = []
    collected_points = 0

    for sampled_batch in data_loader:
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch = volume_batch.to(device)
        label_batch = label_batch.to(device)

        # 前向传播以得到: outputs(logits), embedding(中间特征)
        with torch.no_grad():
            embedding = model(volume_batch)  # 你模型里返回 (outputs,embedding)

        # 对于 2D 分割, embedding shape 通常是 [B, C, H, W]
        # 展平得到 [B*H*W, C]
        b, c, h, w = embedding.shape
        embedding = embedding.permute(0, 2, 3, 1).reshape(-1, c)  # [B*H*W, C]
        labels = label_batch.reshape(-1)                          # [B*H*W]

        embedding = embedding.cpu().numpy()
        labels = labels.cpu().numpy()

        # 如果这一批提取出来的像素总数已经大于剩余可采样的数量，就只取一部分
        remaining = max_points - collected_points
        if embedding.shape[0] > remaining:
            embedding = embedding[:remaining, :]
            labels = labels[:remaining]

        all_features.append(embedding)
        all_labels.append(labels)
        collected_points += embedding.shape[0]

        # 如果已经收集够 max_points 就停止
        if collected_points >= max_points:
            break

    # 如果一张图都没取到，直接返回
    if collected_points == 0:
        print("No data collected for t-SNE visualization.")
        model.train()
        return

    # 拼接所有批次
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # t-SNE 降维 (n_components=2)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate='auto')
    features_2d = tsne.fit_transform(all_features)  # shape: [N, 2]

    # 画图
    fig, ax = plt.subplots()
    sc = ax.scatter(features_2d[:, 0], features_2d[:, 1], c=all_labels, s=2, alpha=0.6)
    ax.set_title(f"t-SNE @iter {iter_num}")
    plt.tight_layout()

    # 把这个可视化图放进 TensorBoard
    writer.add_figure("t-SNE_features", fig, global_step=iter_num)
    plt.close(fig)

    model.train()  # 再切回 train 模式

class MaskConfidence:
    def __init__(self, num_classes):
        # 初始化mIoU计算类
        self.miou_calculator = meanIOU(num_classes)

    def compute_patch_confidence(self, predictions, gts, patch_x_idx, patch_y_idx, patch_size):
        """
        根据每个patch的位置计算其对应的mIoU置信度

        参数:
        - predictions: 模型预测的标签，形状为 (batch_size, num_classes, img_x, img_y)
        - gts: 真实标签，形状为 (batch_size, img_x, img_y)
        - patch_x_idx: 当前patch的x坐标
        - patch_y_idx: 当前patch的y坐标
        - patch_size: 每个patch的大小

        返回:
        - patch_confidence: 当前patch的置信度（mIoU值）
        """
        # 计算当前patch在原图上的位置
        w_start, h_start = patch_x_idx * patch_size, patch_y_idx * patch_size
        w_end, h_end = (patch_x_idx + 1) * patch_size, (patch_y_idx + 1) * patch_size

        # 截取该patch的预测标签和真实标签
        patch_pred = torch.argmax(predictions[:, :, w_start:w_end, h_start:h_end], dim=1)  # shape: (batch_size, img_x, img_y)
        patch_gt = gts[:, w_start:w_end, h_start:h_end]

        # 计算该patch的mIoU
        self.miou_calculator.add_batch(patch_pred, patch_gt)
        iu, _ = self.miou_calculator.evaluate()

        # 取所有类别的最大mIoU值作为该patch的置信度
        patch_confidence = np.max(iu)

        return patch_confidence

def generate_mask(img, predictions, gts, args):
    """
    根据置信度生成掩码，保留高置信度和低置信度的补丁，随机遮盖其他补丁。

    :param img: 输入图像，形状为 (batch_size, channel, img_x, img_y)
    :param predictions: 模型预测的标签，形状为 (batch_size, num_classes, img_x, img_y)
    :param gts: 真实标签，形状为 (batch_size, img_x, img_y)
    :param args: 包含 patch_size 和 mask_density 等参数
    :param preserve_fraction: 保留高低置信度补丁的比例（默认10%）
    :return: mask 和 loss_mask
    """
    batch_size, channel, img_x, img_y = img.shape

    # 创建两个掩码，分别用于返回的掩码和损失计算
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()  # 初始为1
    mask = torch.ones(img_x, img_y).cuda()  # 初始为1

    # 计算每行和每列可以分割的补丁数量
    num_patches_x = img_x // args.patch_size
    num_patches_y = img_y // args.patch_size

    # 计算总补丁数
    total_patches = num_patches_x * num_patches_y

    # 计算需要保留的高置信度和低置信度补丁数
    preserve_patches = int(args.preserve_fraction * total_patches)

    # 计算需要遮盖的补丁数
    num_masked_patches = int(args.mask_density * total_patches)

    # 创建 MaskConfidence 对象，来计算每个补丁的置信度
    mask_confidence = MaskConfidence(num_classes=args.num_classes)

    # 获取所有补丁的置信度
    patch_confidences = {}
    for patch_x_idx in range(num_patches_x):
        for patch_y_idx in range(num_patches_y):
            patch_confidence = mask_confidence.compute_patch_confidence(predictions, gts, patch_x_idx, patch_y_idx, args.patch_size)
            patch_confidences[(patch_x_idx, patch_y_idx)] = patch_confidence

    sorted_patches = sorted(patch_confidences.items(), key=lambda x: x[1])

# 选择置信度最低的 preserve_patches 个补丁作为低置信度补丁
    low_conf_patches = sorted_patches[:preserve_patches]
    # 选择置信度最高的 preserve_patches 个补丁作为高置信度补丁
    high_conf_patches = sorted_patches[-preserve_patches:]

    # 创建一个集合，包含所有需要保留的补丁索引
    preserve_patches_indices = set([idx for idx, conf in low_conf_patches] + [idx for idx, conf in high_conf_patches])

    # 获取可遮盖的补丁索引（即不在保留集合中的补丁）
    eligible_patches = [idx for idx, conf in sorted_patches if idx not in preserve_patches_indices]

    # 计算从可遮盖补丁中需要遮盖的补丁数
    num_mask_from_eligible = min(num_masked_patches, len(eligible_patches))

    # 随机选择需要遮盖的补丁索引
    selected_mask_patches = random.sample(eligible_patches, num_mask_from_eligible)

    # 遍历选择的补丁索引并进行遮盖
    for patch_x_idx, patch_y_idx in selected_mask_patches:
        # 计算当前补丁在原图上的位置
        w_start = patch_x_idx * args.patch_size
        h_start = patch_y_idx * args.patch_size
        w_end = w_start + args.patch_size
        h_end = h_start + args.patch_size

        # 更新掩码
        mask[w_start:w_end, h_start:h_end] = 0
        loss_mask[:, w_start:w_end, h_start:h_end] = 0  # 对应位置的 loss_mask 设为0

    return mask.long(), loss_mask.long()

def obtain_cutmix_box(img,p = 0.5):
    batch_size, channel, img_x, img_y = img.size()
    mask = torch.ones(img_x, img_y).cuda()
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()

    size_min = 0.02
    size_max = 0.4
    ratio_1 = 0.3
    ratio_2 = 1 / 0.3

    if random.random() > p:
        return mask.long(), loss_mask.long()

    size = np.random.uniform(size_min, size_max) * img_x * img_y
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_x)
        y = np.random.randint(0, img_y)

        if x + cutmix_w <= img_x and y + cutmix_h <= img_y:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 0
    loss_mask[:, y:y + cutmix_h, x:x + cutmix_w] = 0

    return mask.long(), loss_mask.long()

def random_mask(img, shrink_param=3):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    x_split, y_split = int(img_x / shrink_param), int(img_y / shrink_param)
    patch_x, patch_y = int(img_x*2/(3*shrink_param)), int(img_y*2/(3*shrink_param))
    mask = torch.ones(img_x, img_y).cuda()
    for x_s in range(shrink_param):
        for y_s in range(shrink_param):
            w = np.random.randint(x_s*x_split, (x_s+1)*x_split-patch_x)
            h = np.random.randint(y_s*y_split, (y_s+1)*y_split-patch_y)
            mask[w:w+patch_x, h:h+patch_y] = 0
            loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
    return mask.long(), loss_mask.long()

def contact_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_y = int(img_y *4/9)
    h = np.random.randint(0, img_y-patch_y)
    mask[h:h+patch_y, :] = 0
    loss_mask[:, h:h+patch_y, :] = 0
    return mask.long(), loss_mask.long()

def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    CE = nn.CrossEntropyLoss(reduction='none')
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
    loss_ce = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16) 
    loss_ce += patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)#loss = loss_ce
    return loss_dice, loss_ce

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

def pre_train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.pre_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path,'{}_best_model.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs/2), int((args.batch_size-args.labeled_bs) / 2)

    model = PRS_net(in_chns=1, class_num=num_classes)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([RandomGenerator(args.image_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path,args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start pre_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            img_mask, loss_mask = generate_mask1(img_a)
            #img_mask,loss_mask = obtain_cutmix_box(img_a, args.cutmix_prob)
            gt_mixl = lab_a * img_mask + lab_b * (1 - img_mask)

            #-- original
            net_input = img_a * img_mask + img_b * (1 - img_mask)
            out_mixl= model(net_input)
            loss_dice, loss_ce = mix_loss(out_mixl, lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True)

            loss = (loss_dice + loss_ce) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            writer.add_scalar('info/mix_ce', loss_ce, iter_num)

            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f'%(iter_num, loss, loss_dice, loss_ce))

            if iter_num % 20 == 0:
                image = net_input[1, 0:1, :, :]
                writer.add_image('pre_train/Mixed_Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(out_mixl, dim=1), dim=1, keepdim=True)
                writer.add_image('pre_train/Mixed_Prediction', outputs[1, ...] * 50, iter_num)
                labs = gt_mixl[1, ...].unsqueeze(0) * 50
                writer.add_image('pre_train/Mixed_GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()

def self_train(args ,pre_snapshot_path, snapshot_path):
    vis_dir = os.path.abspath("vis")
    os.makedirs(vis_dir, exist_ok=True)
    print(f"✅ 图像将保存在: {vis_dir}")
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path,'{}_best_model.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs/2), int((args.batch_size-args.labeled_bs) / 2)

    model = PRS_net(in_chns=1, class_num=num_classes)
    ema_model = PRS_net(in_chns=1, class_num=num_classes, ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([RandomGenerator(args.image_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path,args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    load_net(ema_model, pre_trained_model)
    load_net_opt(model, optimizer, pre_trained_model)
    logging.info("Loaded from {}".format(pre_trained_model))

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start self_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()
    ema_model.train()

    ce_loss = CrossEntropyLoss()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            uimg_a, uimg_b = volume_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch[args.labeled_bs + unlabeled_sub_bs:]
            ulab_a, ulab_b = label_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], label_batch[args.labeled_bs + unlabeled_sub_bs:]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            with torch.no_grad():
                pre_a = ema_model(uimg_a)
                pre_b = ema_model(uimg_b)
                plab_a = get_ACDC_masks(pre_a, nms=1)
                plab_b = get_ACDC_masks(pre_b, nms=1)

                pre_label_a= model(img_a)
                # img_mask, loss_mask = generate_mask1(img_a)
                img_mask, loss_mask = generate_mask(img_a,pre_label_a,lab_a,args)
                # img_mask,loss_mask = obtain_cutmix_box(img_a,args.cutmix_prob)

                unl_label = ulab_a * img_mask + lab_a * (1 - img_mask)
                l_label = lab_b * img_mask + ulab_b * (1 - img_mask)

            consistency_weight = get_current_consistency_weight(iter_num//150)

            lambda_a = (iter_num / max_iterations) *  0.4

            tmp_a = lambda_a*uimg_a + (1-lambda_a)*img_a
            tmp_b = lambda_a*uimg_b + (1-lambda_a)*img_b

            # 进行直方图匹配
            img_a_matched = histogram_match_batch(img_a, tmp_a)
            img_b_matched = histogram_match_batch(img_b, tmp_b)
            # img_a_matched = img_a
            if iter_num % 500 == 0:
                # 增强 tmp_a 作为 reference，提升匹配后的差异性
                exaggerated_tmp_a = torch.clamp(tmp_a * 0.4, 0, 1)  # 降低亮度
                exaggerated_tmp_b = torch.clamp(tmp_b * 1.8, 0, 1)  # 增加亮度
                img_a_matched = histogram_match_batch(img_a, exaggerated_tmp_a)
                img_b_matched = histogram_match_batch(img_b, exaggerated_tmp_b)

                def save_image(tensor_img, filename):
                    img = tensor_img.detach().cpu().numpy()
                    img = np.squeeze(img)
                    img = np.clip(img, 0, 1)   # 限制在0-1范围
                    img = (img * 255).astype(np.uint8)
                    save_path = os.path.join(vis_dir, f"iter{iter_num}_{filename}")
                    success = cv2.imwrite(save_path, img)
                    if success:
                        print(f"✅ 图像保存成功: {save_path}")
                    else:
                        print(f"❌ 图像保存失败: {save_path}")

                # 保存不同版本
                save_image(img_a[0, 0], "img_a.png")
                save_image(tmp_a[0, 0], "tmp_a.png")
                save_image(img_a_matched[0, 0], "img_a_matched.png")

            # net_input_unl = uimg_a * img_mask + img_a * (1 - img_mask)
            # net_input_l = img_b * img_mask + uimg_b * (1 - img_mask)

            #替换组合后的图像
            net_input_unl = uimg_a * img_mask + img_a_matched * (1 - img_mask)
            net_input_l = img_b_matched * img_mask + uimg_b * (1 - img_mask)

            out_unl= model(net_input_unl)
            out_l= model(net_input_l)
            unl_dice, unl_ce = mix_loss(out_unl, plab_a, lab_a, loss_mask, u_weight=args.u_weight, unlab=True)
            l_dice, l_ce = mix_loss(out_l, lab_b, plab_b, loss_mask, u_weight=args.u_weight)

            loss_ce = unl_ce + l_ce
            loss_dice = unl_dice + l_dice

            with torch.no_grad():
                out_unl_ema = ema_model(net_input_unl).detach()
                out_l_ema   = ema_model(net_input_l).detach()

                pseudo_unl_soft = torch.softmax(out_unl_ema, dim=1)
                pseudo_l_soft   = torch.softmax(out_l_ema, dim=1)
                # 此处 pseudo_unl 和 pseudo_l 是类别索引，形状为 [B, H, W]
                confidence_unl, pseudo_unl = torch.max(pseudo_unl_soft, dim=1)
                confidence_l, pseudo_l     = torch.max(pseudo_l_soft, dim=1)
                # 计算 mask_unl 和 mask_l
                mask_unl_orig = (confidence_unl > 0.5).float().unsqueeze(1)
                mask_l_orig   = (confidence_l > 0.5).float().unsqueeze(1)

                # 为逐元素乘法扩展 mask 到与 logits 相同的形状 [B, C, H, W]
            mask_unl_exp = mask_unl_orig.expand_as(out_l)
            mask_l_exp   = mask_l_orig.expand_as(out_unl)

            # 调用 dice_loss 时，将伪标签 unsqueeze 成 [B, 1, H, W]
            pseudo_sup_1 = dice_loss(F.softmax(out_l, dim=1) * mask_unl_exp, pseudo_unl.unsqueeze(1), mask_unl_orig)
            pseudo_sup_2 = dice_loss(F.softmax(out_unl, dim=1) * mask_l_exp, pseudo_l.unsqueeze(1), mask_l_orig)


            pseudo_supervision = pseudo_sup_1 + pseudo_sup_2
            cps_loss = consistency_weight * pseudo_supervision

            loss = (loss_dice + loss_ce) / 2 +   0.1*cps_loss

            # loss = (loss_dice + loss_ce) / 2
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            iter_num += 1
            update_model_ema(model, ema_model, 0.99)

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            writer.add_scalar('info/mix_ce', loss_ce, iter_num)
            writer.add_scalar('info/cps_loss', pseudo_supervision, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)


            # logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f'%(iter_num, loss, loss_dice, loss_ce))
            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f,cps_loss: %f'%(iter_num, loss, loss_dice, loss_ce,pseudo_supervision))

            if iter_num % 20 == 0:
                image = net_input_unl[1, 0:1, :, :]
                writer.add_image('train/Un_Image', image, iter_num)
                image_l = img_a_matched[1, 0:1, :, :]  # 使用直方图匹配后的图像
                # image_l = img_a[1, 0:1, :, :]  # 使用直方图匹配后的图像
                writer.add_image('train/L_Matched_Image', image_l, iter_num)
                outputs = torch.argmax(torch.softmax(out_unl, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Un_Prediction', outputs[1, ...] * 50, iter_num)
                labs = unl_label[1, ...].unsqueeze(0) * 50
                writer.add_image('train/Un_GroundTruth', labs, iter_num)

                image_l = net_input_l[1, 0:1, :, :]
                writer.add_image('train/L_Image', image_l, iter_num)
                outputs_l = torch.argmax(torch.softmax(out_l, dim=1), dim=1, keepdim=True)
                writer.add_image('train/L_Prediction', outputs_l[1, ...] * 50, iter_num)
                labs_l = l_label[1, ...].unsqueeze(0) * 50
                writer.add_image('train/L_GroundTruth', labs_l, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                visualize_tsne(model, trainloader, writer, iter_num, device='cuda', max_points=2000)
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # -- path to save models
    pre_snapshot_path = "/data2/zxm/Projects/PRS/model/ACDC_{}_{}_{}_labeled/pre_train".format(args.exp, args.seed,args.labelnum)
    self_snapshot_path = "/data2/zxm/Projects/PRS/model/ACDC_{}_{}_{}_labeled/self_train".format(args.exp, args.seed,args.labelnum)
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
    shutil.copy('../code/ACDC_PRS_train.py', self_snapshot_path)

    #Pre_train
    logging.basicConfig(filename=pre_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)

    #Self_train
    logging.basicConfig(filename=self_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)

    


