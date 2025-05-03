import os
import time
import logging
import argparse

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torchstat import stat

from util import dataset, transform, config
from util.util import AverageMeter, intersectionAndUnion, check_makedirs, colorize

cv2.ocl.setUseOpenCL(False)

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    # parser.add_argument('--config', type=str, default='config/ade20k/ade20k_pspnet50.yaml', help='config file')
    parser.add_argument('--config', type=str, default='config/cityscapes/cityscapes_pspnet101.yaml', help='config file')
    # parser.add_argument('opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('opts', help='see config/cityscapes/cityscapes_pspnet101.yaml for all options', default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--data_root', type=str, default='dataset/cityscapes')
    parser.add_argument('--train_list', type=str, default='dataset/cityscapes/list/fine_train.txt')
    parser.add_argument('--val_list', type=str, default='dataset/cityscapes/list/fine_val.txt')
    parser.add_argument('--classes', type=int, default=19)

    parser.add_argument('--arch', type=str, default='psp')
    parser.add_argument('--layers', type=int, default=101)
    parser.add_argument('--sync_bn', default=True)              # adopt syncbn or not
    parser.add_argument('--train_h', type=int, default=713)
    parser.add_argument('--train_w', type=int, default=713)
    parser.add_argument('--scale_min', type=float, default=0.5)   # minimum random scale
    parser.add_argument('--scale_max', type=float, default=2.0)
    parser.add_argument('--rotate_min', type=int, default=-10)
    parser.add_argument('--rotate_max', type=int, default=+10)
    parser.add_argument('--zoom_factor', type=int, default=8)
    parser.add_argument('--aux_weight', type=float, default=0.4)
    parser.add_argument('--train_gpu', default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument('--workers', type=int, default=16)      # data loader workers
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--batch_size_val', type=int, default=8)
    parser.add_argument('--base_lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--power', type=float, default=0.9)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--manual_seed')
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='exp/cityscapes/pspnet101/model')
    parser.add_argument('--weight', default=None)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--evaluate',  default=False)
    parser.add_argument('--Distributed')
    parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:6789')
    parser.add_argument('--dist_backend', type=str, default='nccl')
    parser.add_argument('--multiprocessing_distributed', default=True)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)

    # parser.add_argument('--test_list', type=str, default='dataset/cityscapes/list/fine_val.txt')
    parser.add_argument('--test_list', type=str, default='dataset/cityscapes/list/fine_test.txt')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--base_size', type=int, default=2048)
    parser.add_argument('--test_h', type=int, default=713)
    parser.add_argument('--test_w', type=int, default=713)
    parser.add_argument('--scales', default=[1.0])
    parser.add_argument('--has_prediction', default=False)
    parser.add_argument('--index_start', type=int, default=0)
    parser.add_argument('--index_step', type=int, default=0)
    parser.add_argument('--test_gpu', default=[0])
    parser.add_argument('--model_path', type=str, default='exp/cityscapes/pspnet101/model/train_epoch_200.pth')         # evaluation model path
    parser.add_argument('--save_folder', type=str, default='exp/cityscapes/pspnet101/result/epoch_200/val/ss')      # results save folder
    parser.add_argument('--colors_path', type=str, default='data/cityscapes/cityscapes_colors.txt')     # path of dataset colors
    parser.add_argument('--names_path', type=str, default='data/cityscapes/cityscapes_names.txt')       # path of dataset category names



    args = parser.parse_args()
    # assert args.config is not None
    # cfg = config.load_cfg_from_cfg_file(args.config)
    # if args.opts is not None:
    #     cfg = config.merge_cfg_from_list(cfg, args.opts)
    # return cfg
    return args



def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def check(args):
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert args.split in ['train', 'val', 'test']
    if args.arch == 'psp':
        assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0
    elif args.arch == 'psa':
        if args.compact:
            args.mask_h = (args.train_h - 1) // (8 * args.shrink_factor) + 1
            args.mask_w = (args.train_w - 1) // (8 * args.shrink_factor) + 1
        else:
            assert (args.mask_h is None and args.mask_w is None) or (args.mask_h is not None and args.mask_w is not None)
            if args.mask_h is None and args.mask_w is None:
                args.mask_h = 2 * ((args.train_h - 1) // (8 * args.shrink_factor) + 1) - 1
                args.mask_w = 2 * ((args.train_w - 1) // (8 * args.shrink_factor) + 1) - 1
            else:
                assert (args.mask_h % 2 == 1) and (args.mask_h >= 3) and (
                        args.mask_h <= 2 * ((args.train_h - 1) // (8 * args.shrink_factor) + 1) - 1)
                assert (args.mask_w % 2 == 1) and (args.mask_w >= 3) and (
                        args.mask_w <= 2 * ((args.train_h - 1) // (8 * args.shrink_factor) + 1) - 1)
    else:
        raise Exception('architecture not supported yet'.format(args.arch))


def scale_process(model, image, classes, crop_h, crop_w, h, w, mean, std=None, stride_rate=2/3):
    ori_h, ori_w, _ = image.shape
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=mean)
    new_h, new_w, _ = image.shape
    stride_h = int(np.ceil(crop_h*stride_rate))
    stride_w = int(np.ceil(crop_w*stride_rate))
    grid_h = int(np.ceil(float(new_h-crop_h)/stride_h) + 1)
    grid_w = int(np.ceil(float(new_w-crop_w)/stride_w) + 1)
    prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)
    count_crop = np.zeros((new_h, new_w), dtype=float)
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[s_h:e_h, s_w:e_w].copy()
            count_crop[s_h:e_h, s_w:e_w] += 1
            prediction_crop[s_h:e_h, s_w:e_w, :] += net_process(model, image_crop, mean, std)
    prediction_crop /= np.expand_dims(count_crop, 2)
    prediction_crop = prediction_crop[pad_h_half:pad_h_half+ori_h, pad_w_half:pad_w_half+ori_w]
    prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return prediction

def main():
    global args, logger
    args = get_parser()
    args.split = 'test'             #改動
    check(args)
    logger = get_logger()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)
    logger.info(args)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    gray_folder = os.path.join(args.save_folder, 'gray')
    color_folder = os.path.join(args.save_folder, 'color')

    test_transform = transform.Compose([transform.ToTensor()])
    # data_dir = 'testdata'
    test_data = dataset.SemData(split=args.split, data_root=args.data_root, data_list=args.test_list, transform=test_transform)

    #load data
    # data_dir = 'testdata'
    # file_name = os.listdir(data_dir)
    # data_t = []
    # for idx, fn in enumerate(file_name):  # 以idx作为标签如果标签是图片则以另外的函数读取
    #     im = cv2.imread(data_dir + '/' + fn)
    #     im = np.array(im)
    #     data_t.append(im)
    # data_t = np.asarray(data_t)
    # test_dataset = torch.from_numpy(data_t)

    test_dataset = np.random.rand(1,256, 512, 3)
    test_loader = torch.utils.data.DataLoader(test_dataset,  batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
    # del test_dataset

    colors = np.loadtxt(args.colors_path).astype('uint8')
    names = [line.rstrip('\n') for line in open(args.names_path)]
    if not args.has_prediction:
        if args.arch == 'psp':
            from model.pspnet import PSPNet
            model = PSPNet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, pretrained=False)
        elif args.arch == 'psa':
            from model.psanet import PSANet
            model = PSANet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, compact=args.compact,
                           shrink_factor=args.shrink_factor, mask_h=args.mask_h, mask_w=args.mask_w,
                           normalization_factor=args.normalization_factor, psa_softmax=args.psa_softmax, pretrained=False)
        logger.info(model)
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
        if os.path.isfile(args.model_path):
            # logger.info("=> loading checkpoint '{}'".format(args.model_path))
            checkpoint = torch.load(args.model_path)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            # logger.info("=> loaded checkpoint '{}'".format(args.model_path))
        else:
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))

    test(test_loader, test_data.data_list, model, args.classes, mean, std, args.base_size, args.test_h, args.test_w,
             args.scales, gray_folder, color_folder, colors)



def test(test_loader, model, classes, mean, std, base_size, crop_h, crop_w, scales, gray_folder, color_folder, colors):
    # logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    data_time = AverageMeter()
    batch_time = AverageMeter()
    model.eval()
    end = time.time()
    # for i, (input, _) in enumerate(test_loader):
    for i, input in enumerate(test_loader):
        data_time.update(time.time() - end)

        input = np.squeeze(input.numpy(), axis=0)
        # image = np.transpose(input, (1, 2, 0))
        image = input
        h, w, _ = image.shape
        prediction = np.zeros((h, w, classes), dtype=float)
        for scale in scales:
            long_size = round(scale * base_size)
            new_h = long_size
            new_w = long_size
            if h > w:
                new_w = round(long_size/float(h)*w)
            else:
                new_h = round(long_size/float(w)*h)
            image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            prediction += scale_process(model, image_scale, classes, crop_h, crop_w, h, w, mean, std)
        prediction /= len(scales)
        prediction = np.argmax(prediction, axis=2)
        batch_time.update(time.time() - end)
        end = time.time()
        # if ((i + 1) % 10 == 0) or (i + 1 == len(test_loader)):
        #     logger.info('Test: [{}/{}] '
        #                 'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
        #                 'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).'.format(i + 1, len(test_loader),
        #                                                                             data_time=data_time,
        #                                                                             batch_time=batch_time))
        # check_makedirs(gray_folder)
        # check_makedirs(color_folder)
        # gray = np.uint8(prediction)
        # color = colorize(gray, colors)

        # image_path, _ = data_list[i]
        # image_name = image_path.split('/')[-1].split('.')[0]
        # gray_path = os.path.join(gray_folder, image_name + '.png')
        # color_path = os.path.join(color_folder, image_name + '.png')
        # cv2.imwrite(gray_path, gray)
        # color.save(color_path)
    # logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    return prediction




def net_process(model, image, mean, std=None, flip=True):
    input = torch.from_numpy(image.transpose((2, 0, 1))).float()
    if std is None:
        for t, m in zip(input, mean):
            t.sub_(m)
    else:
        for t, m, s in zip(input, mean, std):
            t.sub_(m).div_(s)
    input = input.unsqueeze(0).cuda()
    if flip:
        input = torch.cat([input, input.flip(3)], 0)
    with torch.no_grad():
        output = model(input)
    _, _, h_i, w_i = input.shape
    _, _, h_o, w_o = output.shape
    if (h_o != h_i) or (w_o != w_i):
        output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)
    output = F.softmax(output, dim=1)
    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]
    output = output.data.cpu().numpy()
    output = output.transpose(1, 2, 0)
    return output


def loadmodel(args):
    if not args.has_prediction:
        if args.arch == 'psp':
            from model.pspnet import PSPNet
            model = PSPNet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, pretrained=False)
        elif args.arch == 'psa':
            from model.psanet import PSANet
            model = PSANet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, compact=args.compact,
                           shrink_factor=args.shrink_factor, mask_h=args.mask_h, mask_w=args.mask_w,
                           normalization_factor=args.normalization_factor, psa_softmax=args.psa_softmax,
                           pretrained=False)
        # logger.info(model)
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    if os.path.isfile(args.model_path):
        # logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        # logger.info("=> loaded checkpoint '{}'".format(args.model_path))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))

    return model


def semseg(img, model):
    global args, logger
    args = get_parser()
    args.split = 'test'
    # check(args)
    logger = get_logger()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)
    # logger.info(args)
    # logger.info("=> creating model ...")
    # logger.info("Classes: {}".format(args.classes))

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    gray_folder = os.path.join(args.save_folder, 'gray')
    color_folder = os.path.join(args.save_folder, 'color')

    test_dataset = img          #(1,256, 512, 3)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers,
                                              pin_memory=True)
    colors = np.loadtxt(args.colors_path).astype('uint8')
    # names = [line.rstrip('\n') for line in open(args.names_path)]
    args.scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    # args.scales = [0.5, 1.0, 1.5]
    args.base_size = 512
    output = test(test_loader, model, args.classes, mean, std, args.base_size, args.test_h, args.test_w,
         args.scales, gray_folder, color_folder, colors)

    return output

if __name__ == '__main__':

    args = get_parser()
    img =cv2.imread('lggan.png')
    img = np.expand_dims(img, axis=0)
    model = loadmodel(args)
    output = semseg(img, model)
    gray = np.uint8(output)
    colors = np.loadtxt(args.colors_path).astype('uint8')
    color = colorize(gray, colors)
    color.save('1.png')
    print('end')

    # main()
