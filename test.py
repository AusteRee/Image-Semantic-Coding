from SC_env import SC
from RL_brain import DeepQNetwork
import numpy as np
import torch
import argparse
import os
from os.path import join
import torch.nn as nn
import cv2
import torch.utils.data as data
import torchvision.utils as vutils
from torchvision import models
import math
from predict import semseg, loadmodel, get_parser
# from train import parse
from parses import parse
from SC_env import calculate_iou
from SC_env import calculate_miou
from skimage.metrics import structural_similarity as ssim
# from skimage import io
from pathlib import Path
import imagecodecs
import torch.nn.functional as F
from torchstat import stat
from resprocess import resProcess, pixel_prop, segment_semantic_map

def rgb(img):   #[1,3,256,512]
    rgbimg = torch.zeros(1, 3, 256, 512)
    rgbimg[0, 2, :, :] = img[0, 0, :, :]
    rgbimg[0, 1, :, :] = img[0, 1, :, :]
    rgbimg[0, 0, :, :] = img[0, 2, :, :]
    return rgbimg

def saveimg(img, fn):
    img = rgb(img)
    img = img/256
    vutils.save_image(img, fn)



if __name__ == '__main__':
    args = parse()
    # Device
    device = torch.device('cuda') if args.gpu and torch.cuda.is_available() else torch.device('cpu')
    if args.multi_gpu: assert device.type == 'cuda'

    data_num = 10
    start_index = 0
    resflag = 1


    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    # Load datas
    cityscapes_dir = os.path.join('datasets', 'cityscapes')
    file_name = os.listdir(cityscapes_dir)
    dataset = []
    for idx, fn in enumerate(file_name):
        im = cv2.imread(cityscapes_dir + '/' + fn)
        im = np.array(im)
        dataset.append(im)
    dataset = np.asarray(dataset)
    val_city_dataset = torch.from_numpy(dataset)
    val_city_dataset = val_city_dataset[start_index:start_index+data_num]
    val_citydata = data.DataLoader(val_city_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    del val_city_dataset


    # generate data
    LGGAN_val_dir = os.path.join('datasets', 'synthesized_image')
    file_name = os.listdir(LGGAN_val_dir)
    dataset = []
    for idx, fn in enumerate(file_name):
        im = cv2.imread(LGGAN_val_dir + '/' + fn)
        im = np.array(im)
        dataset.append(im)

    dataset = np.asarray(dataset)
    val_global_dataset = torch.from_numpy(dataset)
    val_global_dataset = val_global_dataset[start_index:start_index+data_num]
    val_LGGANdata = data.DataLoader(val_global_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    del val_global_dataset
    del dataset

    model_vgg = models.vgg16(pretrained=True).to(device)
    model_vgg = nn.Sequential(*list(model_vgg.features)[:-1]).to(device)


    psp_args = get_parser()
    psp_model = loadmodel(psp_args)

    # RL
    env = SC()
    env.t = 0
    RL = DeepQNetwork(env.n_actions, env.n_features, learning_rate=0.01, reward_decay=0.9, e_greedy=1, replace_target_iter=200, memory_size=50, alpha=1)

    RL.q_eval = torch.load('checkpoints/qevamodel' + '.pth')
    # RL.q_eval = model.load_state_dict(torch.load(path), strict=False)
    # RL.q_eval = torch.load('checkpoints/qevamodel' + '.pth')

    # model_path = 'checkpoints/qevamodel'+ str(t) +'.pth'
    # RL.q_eval.load_state_dict(torch.load(model_path))
    # RL.q_target.load_state_dict(torch.load(model_path))
    # RL.q_eval.eval()

    mIou_his = []
    bpp_res = []
    bpp_res_1 = []

    t = 0

    pixes = torch.zeros((data_num, 19))
    bpp_his = torch.zeros((data_num, 19))

    for city, lggan in zip(val_citydata, val_LGGANdata):
        res = city.to(torch.int) - lggan.to(torch.int)
        input_lable = semseg(city.cpu(), psp_model)

        if resflag == 1:
            res = res.numpy()
            res_min = res.min()
            res_max = res.max()
            res, threshold_mask = resProcess(res, input_lable)
        else:
            res = (res + 256) / 2

        res = res.permute(0, 3, 1, 2)
        res = res.squeeze(0)

        lggan = lggan.squeeze(0).permute(2, 0, 1)
        observation = env.reset(input_lable, res)
        output = lggan.unsqueeze(0)

        bpp = 0

        for i in range(19):
            m = observation[2]
            action = RL.choose_action(observation)  # action

            one_hot = torch.ones(256, 512)  # （h, w)
            # one_hot[input_lable_ori != m.item()] = 0  # sm
            one_hot[input_lable != m.item()] = 0  # sm

            pixes[t, i] = one_hot.sum()
            one_hot = one_hot.unsqueeze(0)
            one_hot = torch.cat((one_hot, one_hot, one_hot), 0)
            if resflag == 1:
                observation_, output, bpp_m = env.step_test(action, observation, city, output, model_vgg, psp_model, res,
                                                           input_lable, one_hot, res_max, res_min, threshold_mask)
            else:
                break
            observation = observation_
            bpp_his[t, i] = bpp_m
            bpp += bpp_m

        bpp_res.append(bpp)

        # MIoU
        semseg_output = semseg(output.to(torch.uint8).permute(0, 2, 3, 1).cpu(), psp_model)
        ious_output = calculate_iou(input_lable, semseg_output, 19)
        mIou = calculate_miou(ious_output)
        mIou_his.append(mIou)

        ## save img
        # saveimg(city.permute(0, 3, 1, 2), 'results/' + str(t) + 'city.png')
        # saveimg(lggan.unsqueeze(0), 'results/' + str(t) + 'lggan.png')
        # saveimg(output, 'results/' + str(t) + '.png')

        t += 1

        # print(t)

    print('mIou_his: {:.5f}'.format(sum(mIou_his) / len(mIou_his)))
    print('bpp_res: {:.5f}'.format(sum(bpp_res) / len(bpp_res)))
