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
import logging
from resprocess import resProcess, pixel_prop, segment_semantic_map


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--dataset', type=str, choices=['COCO-Stuff', 'CityScapes-stuff'], default='CityScapes-stuff')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gpu', action='store_true', default='true')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--lr_RL', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--epochs_decay', type=int, default=100)
    # parser.add_argument('--lambdaR', type=float, default=0.001)
    # parser.add_argument('--lambda_Lp', type=float, default=10)
    # parser.add_argument('--lambda_Ls', type=float, default=0.99)
    parser.add_argument('--save_epochs', type=int, default=25)
    parser.add_argument('--experiment_name', type=str, default='cityscapes')
    parser.add_argument('--log_iters', type=int, default=100)
    parser.add_argument('--load_epoch', type=int, default=None)
    parser.add_argument('--n_class', type=int, default=19)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    print(args)

    # Device
    device = torch.device('cuda') if args.gpu and torch.cuda.is_available() else torch.device('cpu')
    if args.multi_gpu: assert device.type == 'cuda'

    data_num = 400
    start_index = 0

    # Load data
    cityscapes_dir = os.path.join('datasets', 'cityscapes')
    file_name = os.listdir(cityscapes_dir)
    dataset = []
    for idx, fn in enumerate(file_name):  #
        im = cv2.imread(cityscapes_dir + '/' + fn)
        im = np.array(im)
        dataset.append(im)
    dataset = np.asarray(dataset)
    train_city_dataset = torch.from_numpy(dataset)
    train_city_dataset = train_city_dataset[start_index:start_index+data_num]
    train_citydata = data.DataLoader(train_city_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    del train_city_dataset

    # generate data
    LGGAN_val_dir = os.path.join('datasets', 'synthesized_image')
    file_name = os.listdir(LGGAN_val_dir)
    dataset = []
    for idx, fn in enumerate(file_name):
        im = cv2.imread(LGGAN_val_dir + '/' + fn)
        im = np.array(im)
        dataset.append(im)
    dataset = np.asarray(dataset)
    train_global_dataset = torch.from_numpy(dataset)
    train_global_dataset = train_global_dataset[start_index:start_index+data_num]
    train_LGGANdata = data.DataLoader(train_global_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    del train_global_dataset
    del dataset

    # lable_dir = os.path.join('datasets', 'output_lable')
    # file_name = os.listdir(lable_dir)
    # dat = []
    # for idx, fn in enumerate(file_name):  #
    #     im = cv2.imread(lable_dir + '/' + fn)
    #     im = cv2.resize(im, [512, 256])
    #     im = np.array(im)
    #     im = im[:, :, 1]
    #     dat.append(im)
    # dat = np.asarray(dat)
    # train_lable_dataset = torch.from_numpy(dat)
    # train_labledata = data.DataLoader(train_lable_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    # del train_lable_dataset

    # Models
    # n_classes = 35
    model_vgg = models.vgg16(pretrained=True).to(device)
    model_vgg = nn.Sequential(*list(model_vgg.features)[:-1]).to(device)

    psp_args = get_parser()
    psp_model = loadmodel(psp_args)

    # RL
    env = SC()
    RL = DeepQNetwork(env.n_actions,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=100,
                      memory_size=10000
                      )

    episodes = 10
    batchSize = 1
    cnt = 0
    t = 0

    # load
    # RL.q_eval = torch.load('checkpoints/' + '.pth')

    lable_dataset = []
    reward_his = []
    resflag = 1
    reward_ep = []
    r_perpic = []

    for i in range(episodes):
        for city, lggan in zip(train_citydata, train_LGGANdata):
            print(t)
            res = city.to(torch.int) - lggan.to(torch.int)

            if i == 0:
                input_lable = semseg(city.cpu(), psp_model)
                lable_dataset.append(input_lable)
            else:
                input_lable = lable_dataset[cnt % data_num]
            # input_lable = lable.squeeze(0)
            # input_lable = input_lable.to(torch.int)
            # input_lable = input_lable.numpy()

            if resflag == 1:
                res = res.numpy()
                res_min = res.min()
                res_max = res.max()
                res, threshold_mask = resProcess(res, input_lable)
            else:
                res = (res + 256) / 2
            # city = city.to(torch.int)
            # lggan = lggan.to(torch.int)

            res = res.permute(0, 3, 1, 2)
            res = res.squeeze(0)
            # res = (res + 256) / 2
            lggan = lggan.squeeze(0).permute(2, 0, 1)

            observation = env.reset(input_lable, res)	 # observation = [sm, fm, m]
            # list {tensor[3, 256, 512], tensor[1, 3, 256, 512], int(m) }
            output = lggan.unsqueeze(0).to(torch.float32)

            for m in range(19):
                m = torch.tensor(m).to(device)
                one_hot = torch.ones(256, 512)  # （h, w)
                one_hot[input_lable != m.item()] = 0  # sm
                one_hot = one_hot.unsqueeze(0)
                one_hot = torch.cat((one_hot, one_hot, one_hot), 0)
                action = RL.choose_action(observation)		# action	tensor([2], device='cuda:0')

                # observation_, reward, output = env.step(action, observation, city, output, model_vgg,
                #                                         psp_model, res, input_lable, one_hot, res_min, res_max, cnt)
                observation_, reward, output = env.stepTrain(action, observation, city, output, model_vgg,
                                                                    psp_model, res, input_lable, one_hot, res_min,
                                                                    res_max, cnt, lggan, threshold_mask)

                RL.store_transition(observation, action, reward, observation_)
                # for _ in range(batchSize):
                #     RL.learn()
                RL.learn()

                observation = observation_
                t += 1

                # if t % 1900 == 0:
                #     torch.save(RL.q_eval, 'checkpoints/0410/qevamodel0410' + str(t) + str(env.beta) + '.pth')

            cnt += 1
            # reward_his.append(reward_pic)
            # reward_his += reward_pic
            # print(reward_pic)


        torch.save(RL.q_eval, 'checkpoints/' + str(i) + '.pth')
    # torch.save(RL.q_eval.state_dict(), join(checkpoint_path, '{:03}.model.pth'.format(t)))
    # torch.save(model_opt.state_dict(), join(checkpoint_path, '{:03}.model_opt.pth'.format(ep)))

    # print(RL.q_eval)
    # print(RL.q_eval.state_dict())
    torch.save(RL.q_eval, 'checkpoints/' + '.pth')

    logging.basicConfig(
        filename='reward_log.log',
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

# RL.plot_cost()

