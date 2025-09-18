from Net.CMMDL import CMMDL
import os
import torch
import torch.nn as nn
import warnings
from torch.utils.data import DataLoader
from PIL import Image
from TaskFusion_dataset import Fusion_dataset
from Evaluator import *
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 使用 Agg 后端
warnings.filterwarnings("ignore")

def RGB2YCrCb(rgb_image):
    """
    将RGB格式转换为YCrCb格式
    用于中间结果的色彩空间转换中,因为此时rgb_image默认size是[B, C, H, W]
    :param rgb_image: RGB格式的图像数据
    :return: Y, Cr, Cb
    """

    R = rgb_image[:, 0:1]
    G = rgb_image[:, 1:2]
    B = rgb_image[:, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = Y.clamp(0.0,1.0)
    Cr = Cr.clamp(0.0,1.0).detach()
    Cb = Cb.clamp(0.0,1.0).detach()
    return Y, Cb, Cr

def YCbCr2RGB(Y, Cb, Cr):
    """
    将YcrCb格式转换为RGB格式
    :param Y:
    :param Cb:
    :param Cr:
    :return:
    """
    ycrcb = torch.cat([Y, Cr, Cb], dim=1)
    B, C, W, H = ycrcb.shape
    im_flat = ycrcb.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3)
    out = out.clamp(0,1.0)
    return out

def image_read_cv2(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img

def test(fusion_model, fused_dir, ori_image_folder):
    fusion_model_path = fusion_model
    fused_dir = fused_dir
    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
    fusion_model = nn.DataParallel(CMMDL()).to(device)
    fusion_model.load_state_dict(torch.load(fusion_model_path, map_location='cuda:0'))
    fusion_model.eval()
    IR_path = ori_image_folder + '/infrared'
    VIS_path = ori_image_folder + '/visible'

    test_dataset = Fusion_dataset('val', ir_path=IR_path, vi_path=VIS_path)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    with torch.no_grad():
        for (images_vis, images_ir, name) in tqdm(test_loader, desc='Processing Images'):
            images_vis = images_vis.to(device)
            images_ir = images_ir.to(device)
            Y, Cb, Cr = RGB2YCrCb(images_vis)
            images_fuse = fusion_model(Y, images_ir)
            fusion_image = YCbCr2RGB(images_fuse, Cb, Cr)
            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)
            fused_image = fusion_image.cpu().numpy()
            fused_image = fused_image.transpose((0, 2, 3, 1))
            fused_image = (fused_image - np.min(fused_image)) / (
                    np.max(fused_image) - np.min(fused_image)
            )
            fused_image = np.uint8(255.0 * fused_image)
            for k in range(len(name)):
                image = fused_image[k, :, :, :]
                image = Image.fromarray(image)
                save_path = os.path.join(fused_dir, name[k])
                image.save(save_path)

def evaluate(eval_folder, ori_image_folder):
    eval_folder = eval_folder
    ori_image_folder = ori_image_folder
    metric_result = np.zeros((12))
    model_name = 'CMMDL'
    image_list = os.listdir(os.path.join(ori_image_folder, "ir"))
    headers = ['EN', 'SD', 'SF', 'MI', 'SCD', 'VIF', 'Qabf', 'SSIM', 'AG', 'MSE', 'CC', 'PSNR']
    metric_count = len(headers)
    for img_name in tqdm(image_list, desc='Processing Images'):
        ir = image_read_cv2(os.path.join(ori_image_folder, "infrared", img_name), 'GRAY')
        vi = image_read_cv2(os.path.join(ori_image_folder, "visible", img_name), 'GRAY')
        fi = image_read_cv2(os.path.join(eval_folder, img_name.split('.')[0] + ".png"), 'GRAY')
        metric_result += np.array([Evaluator.EN(fi), Evaluator.SD(fi)
                                      , Evaluator.SF(fi), Evaluator.MI(fi, ir, vi)
                                      , Evaluator.SCD(fi, ir, vi), Evaluator.VIFF(fi, ir, vi)
                                      , Evaluator.Qabf(fi, ir, vi), Evaluator.SSIM(fi, ir, vi)
                                      , Evaluator.AG(fi), Evaluator.MSE(fi, ir, vi), Evaluator.CC(fi, ir, vi)
                                      , Evaluator.PSNR(fi, ir, vi)])
    metric_result /= len(os.listdir(eval_folder))
    col_widths = [max(len(header), len(f"{metric_result[i]:.2f}")) for i, header in enumerate(headers)]
    header_row = ' ' * len(model_name) + '\t' + '\t'.join(f"{header:^{col_widths[i]}}" for i, header in enumerate(headers))
    print(header_row)
    result_row = model_name + '\t' + '\t'.join(f"{metric_result[i]:^{col_widths[i]}.2f}" for i in range(metric_count))
    print(result_row)
    print("=" * (len(header_row) + metric_count))

def main():
    fusion_model_path = './Model/CMMDL_model_finally.pth'
    fused_dir = './result'
    ori_image_folder = './test_cases/'
    os.makedirs(fused_dir, mode=0o777, exist_ok=True)
    test(fusion_model_path, fused_dir, ori_image_folder)
    evaluate(fused_dir, ori_image_folder)

if __name__ == '__main__':

    main()
