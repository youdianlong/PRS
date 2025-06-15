import argparse
import os
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm
import matplotlib.pyplot as plt

from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data2/zxm/Projects/PRS/data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='PRSb', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4, help='output channel of network')
parser.add_argument('--labelnum', type=int, default=7, help='labeled data')
parser.add_argument('--stage_name', type=str, default='self_train', help='self or pre')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, jc, hd95, asd

def apply_segmentation_overlay(base_img, mask, alpha=0.5):
    """
    将多类别 mask (值为 0,1,2,3...) 叠加到灰度图像上，采用 RGB 分色。
    - base_img: [H,W] 灰度图
    - mask: [H,W] 掩码图（值 0 为背景，其余为类别）
    - alpha: 透明度
    """
    # 归一化灰度图像
    img_rgb = np.stack([base_img]*3, axis=-1)
    img_rgb = img_rgb / img_rgb.max()

    # 定义颜色（背景为0，分类从1开始）
    color_dict = {
        1: [1, 0, 0],  # Red
        2: [0, 1, 0],  # Green
        3: [0, 0, 1],  # Blue
        # 可以添加更多类别
    }

    color_mask = np.zeros_like(img_rgb)

    for label_val, color in color_dict.items():
        mask_area = mask == label_val
        for c in range(3):
            color_mask[:, :, c][mask_area] = color[c]

    overlay = (1 - alpha) * img_rgb + alpha * color_mask
    overlay = np.clip(overlay, 0, 1)
    return overlay

def save_visualization(case_name, save_dir):
    img_path = os.path.join(save_dir, f"{case_name}_img.nii.gz")
    gt_path = os.path.join(save_dir, f"{case_name}_gt.nii.gz")
    pred_path = os.path.join(save_dir, f"{case_name}_pred.nii.gz")

    # 使用 nibabel 读取
    img = nib.load(img_path).get_fdata()
    gt = nib.load(gt_path).get_fdata()
    pred = nib.load(pred_path).get_fdata()

    # 取中间层
    slice_idx = img.shape[2] // 2
    # slice_img = img[:, :, slice_idx]
    # slice_gt = gt[:, :, slice_idx]
    # slice_pred = pred[:, :, slice_idx]
    #
    # # 可视化图像
    # fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    #
    # axs[0].imshow(slice_img, cmap='gray')
    # axs[0].set_title('Input')
    # axs[0].axis('off')
    #
    # axs[1].imshow(slice_img, cmap='gray')
    # axs[1].imshow(slice_pred, cmap='jet', alpha=0.5)
    # axs[1].set_title('Prediction')
    # axs[1].axis('off')
    #
    # axs[2].imshow(slice_img, cmap='gray')
    # axs[2].imshow(slice_gt, cmap='jet', alpha=0.5)
    # axs[2].set_title('Ground Truth')
    # axs[2].axis('off')

    slice_img = img[:, :, slice_idx]
    slice_pred = pred[:, :, slice_idx]
    slice_gt = gt[:, :, slice_idx]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(slice_img, cmap='gray')
    axs[0].set_title("Input Image")
    axs[0].axis('off')

    axs[1].imshow(apply_segmentation_overlay(slice_img, slice_pred))
    axs[1].set_title("Prediction")
    axs[1].axis('off')

    axs[2].imshow(apply_segmentation_overlay(slice_img, slice_gt))
    axs[2].set_title("Ground Truth")
    axs[2].axis('off')

    plt.tight_layout()
    # plt.savefig('vis_case_overlay.png', dpi=300)

    os.makedirs(os.path.join(save_dir, 'vis'), exist_ok=True)
    out_path = os.path.join(save_dir, 'vis', f'{case_name}_vis.png')
    plt.savefig(out_path, dpi=300)
    plt.close()

# def visualize_results(case, image, label, prediction, first_metric, second_metric, third_metric, test_save_path):
#     """
#     可视化原始图像、标签和预测结果，并显示计算的指标
#     """
#     # 创建一个1行4列的图
#     fig, ax = plt.subplots(1, 4, figsize=(20, 5))
#
#     # 原始图像
#     ax[0].imshow(image[0, :, :], cmap='gray')
#     ax[0].set_title("Original Image")
#     ax[0].axis('off')
#
#     # 真实标签
#     ax[1].imshow(label[0, :, :], cmap='jet', alpha=0.5)
#     ax[1].set_title("Ground Truth")
#     ax[1].axis('off')
#
#     # 预测结果
#     ax[2].imshow(prediction[0, :, :], cmap='jet', alpha=0.5)
#     ax[2].set_title("Prediction")
#     ax[2].axis('off')
#
#     # 显示计算的评价指标
#     ax[3].axis('off')
#     ax[3].text(0.1, 0.8, f"First Metric: Dice={first_metric[0]:.4f}, Jaccard={first_metric[1]:.4f}, ASD={first_metric[3]:.4f}, HD95={first_metric[2]:.4f}", fontsize=12)
#     ax[3].text(0.1, 0.6, f"Second Metric: Dice={second_metric[0]:.4f}, Jaccard={second_metric[1]:.4f}, ASD={second_metric[3]:.4f}, HD95={second_metric[2]:.4f}", fontsize=12)
#     ax[3].text(0.1, 0.4, f"Third Metric: Dice={third_metric[0]:.4f}, Jaccard={third_metric[1]:.4f}, ASD={third_metric[3]:.4f}, HD95={third_metric[2]:.4f}", fontsize=12)
#     ax[3].set_title("Metrics")
#
#     # 保存可视化结果
#     plt.savefig(os.path.join(test_save_path, f"{case}_visualization.png"))
#     plt.close(fig)

def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out_main = net(input)
            if len(out_main)>1:
                out_main=out_main[0]
            out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred
    if np.sum(prediction == 1)==0:
        first_metric = 0,0,0,0
    else:
        first_metric = calculate_metric_percase(prediction == 1, label == 1)

    if np.sum(prediction == 2)==0:
        second_metric = 0,0,0,0
    else:
        second_metric = calculate_metric_percase(prediction == 2, label == 2)

    if np.sum(prediction == 3)==0:
        third_metric = 0,0,0,0
    else:
        third_metric = calculate_metric_percase(prediction == 3, label == 3)

    # #可视化结果
    # visualize_results(case, image, label, prediction, first_metric, second_metric, third_metric, test_save_path)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric


def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])
    snapshot_path = "/data2/zxm/Projects/PRS/model/ACDC_{}_{}_{}_labeled/{}".format(FLAGS.exp,FLAGS.seed, FLAGS.labelnum, FLAGS.stage_name)
    test_save_path = "/data2/zxm/Projects/PRS/model/ACDC_{}_{}_{}_labeled/{}_predictions/".format(FLAGS.exp,FLAGS.seed, FLAGS.labelnum, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes)
    save_model_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_model_path, weights_only=True))

    print("init weight from {}".format(save_model_path))
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume(case, net, test_save_path, FLAGS)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
        # 保存可视化图像
        save_visualization(case, test_save_path)

    avg_metric = [first_total / len(image_list), second_total / len(image_list), third_total / len(image_list)]
    return avg_metric, test_save_path


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric, test_save_path = Inference(FLAGS)
    print(metric)
    print((metric[0]+metric[1]+metric[2])/3)
    with open(test_save_path+'../performance.txt', 'w') as f:
        f.writelines('metric is {} \n'.format(metric))
        f.writelines('average metric is {}\n'.format((metric[0]+metric[1]+metric[2])/3))
