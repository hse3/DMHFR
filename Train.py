import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.xxt import xxt
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
import logging
import gc
import shutil
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import convolve
from torchstat import stat
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis

# import matplotlib.pyplot as plt

_EPS = np.spacing(1)


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def _prepare_data(pred: np.ndarray, gt: np.ndarray) -> tuple:
    gt = gt > 128
    pred = pred / 255
    if pred.max() != pred.min():
        pred = (pred - pred.min()) / (pred.max() - pred.min())  # 最大值变成1，最小值变成0
    return pred, gt


def cal_mae(pred: np.ndarray, gt: np.ndarray, area) -> float:
    pred, gt = _prepare_data(pred, gt)
    if area is not None:
        mae = np.sum(np.abs(pred - gt)) / np.sum(area)
    else:
        mae = np.mean(np.abs(pred - gt))
    return mae


def cal_iou(pred: np.ndarray, target: np.ndarray):
    """原来的写法"""
    Iand1 = np.sum(target * pred)
    Ior1 = np.sum(target) + np.sum(pred) - Iand1
    iou_score = Iand1 / (Ior1 + _EPS)
    "网上的写法"
    # pred, target = _prepare_data(pred, target)
    # intersection = pred * target  # 计算交集  pred ∩ true
    # temp = pred + target  # pred + true
    # union = temp - intersection  # 计算并集：A ∪ B = A + B - A ∩ B
    # smooth = 1  # 防止分母为 0
    # iou_score = (intersection.sum() + smooth) / (union.sum() + smooth)
    "网上的写法, 01硬标签版本"
    # pred = np.where(pred > 0.5, 1, 0)
    # intersection = pred * target  # 计算交集  pred ∩ true
    # temp = pred + target  # pred + true
    # union = temp - intersection  # 计算并集：A ∪ B = A + B - A ∩ B
    # smooth = 1  # 防止分母为 0
    # iou_score = (intersection.sum() + smooth) / (union.sum() + smooth)
    return iou_score


def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


def cal_wfm(pred: np.ndarray, gt: np.ndarray) -> float:
    E = np.abs(pred - gt)
    dst, idst = distance_transform_edt(1 - gt, return_indices=True)

    K = fspecial_gauss(7, 5)
    Et = E.copy()
    Et[gt != 1] = Et[idst[:, gt != 1][0], idst[:, gt != 1][1]]
    EA = convolve(Et, K, mode='nearest')
    MIN_E_EA = E.copy()
    MIN_E_EA[(gt == 1) & (EA < E)] = EA[(gt == 1) & (EA < E)]

    B = np.ones_like(gt)
    B[gt != 1] = 2.0 - 1 * np.exp(np.log(1 - 0.5) / 5 * dst[gt != 1])
    Ew = MIN_E_EA * B

    TPw = np.sum(gt) - np.sum(Ew[gt == 1])
    FPw = np.sum(Ew[gt != 1])

    R = 1 - np.mean(Ew[gt == 1])
    P = TPw / (TPw + FPw + np.finfo(np.float64).eps)
    Q = 2 * R * P / (R + P + np.finfo(np.float64).eps)

    return Q


def centroid(matrix: np.ndarray) -> tuple:
    h, w = matrix.shape
    if matrix.sum() == 0:
        x = np.round(w / 2)
        y = np.round(h / 2)
    else:
        area_object = np.sum(matrix)
        row_ids = np.arange(h)
        col_ids = np.arange(w)
        x = np.round(np.sum(np.sum(matrix, axis=0) * col_ids) / area_object)
        y = np.round(np.sum(np.sum(matrix, axis=1) * row_ids) / area_object)
    return int(x) + 1, int(y) + 1


def divide_with_xy(pred: np.ndarray, gt: np.ndarray, x, y) -> dict:
    h, w = gt.shape
    area = h * w

    gt_LT = gt[0:y, 0:x]
    gt_RT = gt[0:y, x:w]
    gt_LB = gt[y:h, 0:x]
    gt_RB = gt[y:h, x:w]

    pred_LT = pred[0:y, 0:x]
    pred_RT = pred[0:y, x:w]
    pred_LB = pred[y:h, 0:x]
    pred_RB = pred[y:h, x:w]

    w1 = x * y / area
    w2 = y * (w - x) / area
    w3 = (h - y) * x / area
    w4 = 1 - w1 - w2 - w3

    return dict(gt=(gt_LT, gt_RT, gt_LB, gt_RB),
                pred=(pred_LT, pred_RT, pred_LB, pred_RB),
                weight=(w1, w2, w3, w4))


def ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    h, w = pred.shape
    N = h * w

    x = np.mean(pred)
    y = np.mean(gt)

    sigma_x = np.sum((pred - x) ** 2) / (N - 1)
    sigma_y = np.sum((gt - y) ** 2) / (N - 1)
    sigma_xy = np.sum((pred - x) * (gt - y)) / (N - 1)

    alpha = 4 * x * y * sigma_xy
    beta = (x ** 2 + y ** 2) * (sigma_x + sigma_y)

    if alpha != 0:
        score = alpha / (beta + _EPS)
    elif alpha == 0 and beta == 0:
        score = 1
    else:
        score = 0
    return score


def region(pred: np.ndarray, gt: np.ndarray) -> float:
    x, y = centroid(gt)
    part_info = divide_with_xy(pred, gt, x, y)
    w1, w2, w3, w4 = part_info['weight']
    pred1, pred2, pred3, pred4 = part_info['pred']
    gt1, gt2, gt3, gt4 = part_info['gt']
    score1 = ssim(pred1, gt1)
    score2 = ssim(pred2, gt2)
    score3 = ssim(pred3, gt3)
    score4 = ssim(pred4, gt4)

    return w1 * score1 + w2 * score2 + w3 * score3 + w4 * score4


def s_object(pred: np.ndarray, gt: np.ndarray) -> float:
    x = np.mean(pred[gt == 1])
    sigma_x = np.std(pred[gt == 1])
    score = 2 * x / (np.power(x, 2) + 1 + sigma_x + _EPS)
    return score


def object(pred: np.ndarray, gt: np.ndarray) -> float:
    fg = pred * gt
    bg = (1 - pred) * (1 - gt)
    u = np.mean(gt)
    object_score = u * s_object(fg, gt) + (1 - u) * s_object(bg, 1 - gt)
    return object_score


def cal_sm(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = _prepare_data(pred, gt)
    y = np.mean(gt)
    if y == 0:
        sm = 1 - np.mean(pred)
    elif y == 1:
        sm = np.mean(pred)
    else:
        sm = 0.5 * object(pred, gt) + 0.5 * region(pred, gt)
        sm = max(0, sm)
    return sm

def _get_adaptive_threshold(matrix: np.ndarray, max_value: float = 1) -> float:
    return min(2 * matrix.mean(), max_value)

def cal_em(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = _prepare_data(pred=pred, gt=gt)
    gt_fg_numel = np.count_nonzero(gt)
    gt_size = gt.shape[0] * gt.shape[1]
    threshold = _get_adaptive_threshold(pred, max_value=1)

    binarized_pred = pred >= threshold
    fg_fg_numel = np.count_nonzero(binarized_pred & gt)
    fg_bg_numel = np.count_nonzero(binarized_pred & ~gt)

    fg___numel = fg_fg_numel + fg_bg_numel
    bg___numel = gt_size - fg___numel

    if gt_fg_numel == 0:
        enhanced_matrix_sum = bg___numel
    elif gt_fg_numel == gt_size:
        enhanced_matrix_sum = fg___numel
    else:
        parts_numel, combinations = generate_parts_numel_combinations(
            fg_fg_numel=fg_fg_numel, fg_bg_numel=fg_bg_numel,
            pred_fg_numel=fg___numel, pred_bg_numel=bg___numel,
        )

        results_parts = []
        for i, (part_numel, combination) in enumerate(zip(parts_numel, combinations)):
            align_matrix_value = 2 * (combination[0] * combination[1]) / \
                                    (combination[0] ** 2 + combination[1] ** 2 + _EPS)
            enhanced_matrix_value = (align_matrix_value + 1) ** 2 / 4
            results_parts.append(enhanced_matrix_value * part_numel)
        enhanced_matrix_sum = sum(results_parts)

    em = enhanced_matrix_sum / (gt_size - 1 + _EPS)
    return em



def test(model, path, dataset):
    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)
    DSC: float = 0.0

    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res, res1, res2, res3 = model(image)
        # eval Dice
        res = F.upsample(res + res1 + res2 + res3, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = res
        target = np.array(gt)
        pred = res
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        # dice = format(dice, '.4f')
        dice = float(dice)

        DSC = DSC + dice
        iou_score = cal_iou(pred, target)
        wfm = cal_wfm(pred, gt)
        smeasure = cal_sm(pred, target)
        meanEm = cal_em(pred, target)
        mae = cal_mae(pred, target, None)

    return DSC / num1, iou_score,  wfm, smeasure, meanEm, mae
    # return DSC / num1


def train(train_loader, model, optimizer, epoch, test_path):
    model.train()
    global best
    size_rates = [0.75, 1, 1.25]
    loss_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            P1, P2, P3, P4 = model(images)
            # ---- loss function ----
            loss_P1 = structure_loss(P1, gts)
            loss_P2 = structure_loss(P2, gts)
            loss_P3 = structure_loss(P3, gts)
            loss_P4 = structure_loss(P4, gts)
            loss = loss_P1 + loss_P2 + loss_P3 + loss_P4
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record.show()))

    # print('\n')

    # save model
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), save_path + str(epoch) + 'xxt.pth')
    # choose the best model

    # global dict_plot

    test1path = './dataset/TestDataset/'
    if (epoch + 1) % 1 == 0:
        for dataset in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
            dataset_dice, dataset_iou, ds_wfm, ds_sm, ds_meanEm, dataset_mae = test(model, test1path, dataset)
            # dataset_dice = test(model, test1path, dataset)

            dataset_dice = format(dataset_dice, '.4f')
            dataset_iou = format(dataset_iou, '.4f')
            ds_wfm = format(ds_wfm, '.4f')
            ds_sm = format(ds_sm, '.4f')
            ds_meanEm = format(ds_meanEm, '.4f')
            dataset_mae = format(dataset_mae, '.4f')

            logging.info(
                'epoch: {}, dataset: {}, dice: {}, iou: {}, wfm: {}, sm: {}, meanEm: {}, mae: {}'.format(epoch, dataset,
                                                                                                         dataset_dice, dataset_iou, ds_wfm, ds_sm, ds_meanEm, dataset_mae))

            # logging.info(
            #     'epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, dataset_dice))

            print(dataset, 'dice: ', dataset_dice, '\t\t\tiou: ', dataset_iou, 'wfm: ', ds_wfm, 'ds_sm: ', ds_sm, 'ds_meanEm: ', ds_meanEm, 'mae: ', dataset_mae)

            # print(dataset, 'dice: ', dataset_dice)

            # dict_plot[dataset].append(dataset_dice)

        meandice, iou, wfm, smea, meanEm, mae, = test(model, test_path, 'test')

        # meandice = test(model, test_path, 'test')

        # dict_plot['test'].append(meandice)

        if meandice > best:
            shutil.rmtree(save_path)
            os.mkdir(save_path)
            best = meandice
            torch.save(model.state_dict(), save_path + 'xxt.pth')
            torch.save(model.state_dict(), save_path + str(epoch) + 'xxt-best.pth')
            print('##############################################################################best', best)
            logging.info(
                '##############################################################################best:{}'.format(best))
        else:
            os.remove(save_path + str(epoch) + 'xxt.pth')

if __name__ == '__main__':
    # dict_plot = {'CVC-300': [], 'CVC-ClinicDB': [], 'Kvasir': [], 'CVC-ColonDB': [], 'ETIS-LaribPolypDB': [],
    #              'test': []}
    name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test']
    ##################model_name#############################
    model_name = 'xxt'
    ###############################################
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=200, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='Adam', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation',
                        default=False, help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')

    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--train_path', type=str,
                        default='./dataset/TrainDataset',
                        help='path to train dataset')

    parser.add_argument('--test_path', type=str,
                        default='./dataset/TestDataset/',
                        help='path to testing Kvasir dataset')

    parser.add_argument('--train_save', type=str,
                        default='./model_pth' + model_name + '/')

    opt = parser.parse_args()
    logging.basicConfig(filename='_1st_transxxt_train_log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    # model = xxt().cuda()
    from thop import profile
    model = xxt().cuda()
    # summary(model, (1, 3, 352, 352))
    inputss = torch.randn(1, 3, 512, 512).cuda()
    macs, params = profile(model, inputs=(inputss,))
    print('Flops: % .4fG'%(macs / 1000000000))

    best = 0

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=opt.lr, weight_decay=1e-4)
    elif opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=opt.lr)
    else:
        optimizer = torch.optim.SGD(params=model.parameters(), lr=opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)
    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=2, trainsize=opt.trainsize,
                              augmentation=opt.augmentation)
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
        # adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
        train(train_loader, model, optimizer, epoch, opt.test_path)

    # plot the eval.png in the training stage
    # plot_train(dict_plot, name)
