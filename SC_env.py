import numpy as np
import torch
import torch.nn as nn
from predict import semseg
import math
import torchvision.utils as vutils
import imagecodecs

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

def calculate_iou(gt_masks, pred_masks, num_classes, print_c_i=False):
    ious = []
    for class_id in range(num_classes):
        gt_mask = gt_masks == class_id
        pred_mask = pred_masks == class_id
        intersection = np.logical_and(gt_mask, pred_mask)
        union = np.logical_or(gt_mask, pred_mask)
        iou = np.sum(intersection) / np.sum(union)

        if print_c_i:
            print(f'class_id:{class_id}, iou:{iou}')

        ious.append(iou)

    return ious

def calculate_miou(ious):
    countt = 0
    summ = 0
    for i in ious:
        if not math.isnan(i) :  # false:
            countt += 1
            summ += i
    miou = summ / countt
    return miou


class SC():

    def __init__(self):
        super(SC, self).__init__()
        self.action_space = [0, 99, 90, 80, 60]
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.device = torch.device('cuda')
        self.l1_norm = nn.L1Loss()
        self.l2_norm = nn.MSELoss()
        self.lamda = 20
        self.alpha = 0.01
        self.beta = 2.5


    def reset(self, input_lable, res):
        m = torch.tensor(0)
        one_hot = torch.ones(256, 512)  # （h, w)
        one_hot[input_lable != m.item()] = 0
        # one_hot = one_hot.view(1, 256, 512)
        sm = one_hot.unsqueeze(0)
        sm = torch.cat((sm, sm, sm), 0)  # [3, 256, 512]
        fm = sm * res
        observation = [sm, fm, m]

        return observation

    def stepTrain(self, action, observation, city, output, model_vgg, psp_model,
                         res, input_lable, one_hots, res_max, res_min, cnt, lggan, threshold_mask):
        # sm = observation[0]
        fm = observation[1]
        m = torch.tensor([observation[2]])

        if action == 0:
            Q = 0
        elif action == 1:
            Q = 99
        elif action == 2:
            Q = 90
        elif action == 3:
            Q = 80
        elif action == 4:
            Q = 60
        else:
            Q = 0

        fm = fm * one_hots
        if Q == 0:
            res_quan = fm * 0
            bpp_xl = 0
        else:
            res_xl = imagecodecs.jpegxl_encode(fm.permute(1, 2, 0).to(torch.uint8).numpy(), level=Q)
            res_xl_size = len(res_xl)
            bpp_xl = res_xl_size * 8 / (256 * 512)
            res_xl_decode = imagecodecs.jpegxl_decode(res_xl)
            res_xl_decode = torch.from_numpy(res_xl_decode)
            res_xl_decode = res_xl_decode.to(int)
            res_xl_decode = res_xl_decode * 2 - 256
            res_quan = res_xl_decode.permute(2, 0, 1) * one_hots
            mask_tensor = torch.from_numpy(threshold_mask).permute(0, 3, 1, 2).squeeze(0)
            res_quan = torch.where(mask_tensor, torch.tensor(0, dtype=res_quan.dtype), res_quan)


        if one_hots.sum() == 0:
            reward = 0
            output1 = lggan.unsqueeze(0).to(torch.float32)
        else:
            output1 = lggan.unsqueeze(0).to(torch.float32) + res_quan.unsqueeze(0)
            output = lggan.unsqueeze(0).to(torch.float32)

            # Lp 感知loss
            feature_real = model_vgg(city.to(torch.float32).permute(0, 3, 1, 2).to(self.device)).detach() #VGG
            feature_1 = model_vgg(output1.to(torch.float32).to(self.device)).detach()
            lp1 = self.l2_norm(feature_real, feature_1)
            feature_0 = model_vgg(output.to(torch.float32).to(self.device)).detach()
            lp0 = self.l2_norm(feature_real, feature_0)

            # Ls 语义loss
            semseg0 = semseg(output.permute(0, 2, 3, 1).to(torch.uint8).cpu(), psp_model)
            if Q == 0:
                semseg1 = semseg0
            else:
                semseg1 = semseg(output1.permute(0, 2, 3, 1).to(torch.uint8).cpu(), psp_model)

            ious0 = calculate_iou(input_lable, semseg0, 19)
            ls0 = calculate_miou(ious0)
            ious1 = calculate_iou(input_lable, semseg1, 19)
            ls1 = calculate_miou(ious1)

            # reward
            Lm = self.lamda * (1-ls0) + self.alpha * lp0
            Lm1 = self.lamda * (1-ls1) + self.alpha * lp1 + self.beta * bpp_xl
            reward = Lm - Lm1
            # print('reward:', reward)

        # update observation
        one_hot1 = torch.ones(256, 512)  # （h, w)
        m += 1
        one_hot1[input_lable != m.item()] = 0  # sm语义图
        sm1 = one_hot1.unsqueeze(0)
        sm1 = torch.cat((sm1, sm1, sm1), 0)  # [3, 256, 512]
        fm1 = one_hot1 * res
        observation_ = [sm1, fm1, m]        #state m+1 observation[sm, fm, m+1]

        return observation_, reward, output1




    def step_test(self, action, observation, city, output, model_vgg, psp_model, res, input_lable, one_hots,
                 res_max, res_min, threshold_mask):
        # sm = observation[0]
        fm = observation[1]
        m = torch.tensor([observation[2]])
        if action == 0:
            Q = 0
        elif action == 1:
            Q = 99
        elif action == 2:
            Q = 90
        elif action == 3:
            Q = 80
        elif action == 4:
            Q = 60
        else:
            Q = 0
        fm = fm * one_hots

        if Q == 0:
            res_quan = fm * 0
            bpp_xl = 0
            res_xl_size = 0

        else:
            res_xl = imagecodecs.jpegxl_encode(fm.permute(1, 2, 0).to(torch.uint8).numpy(), level=Q)
            res_xl_size = len(res_xl)
            bpp_xl = res_xl_size / (256 * 512)

            res_xl_decode = imagecodecs.jpegxl_decode(res_xl)
            res_xl_decode = torch.from_numpy(res_xl_decode)
            res_xl_decode = res_xl_decode.to(int)
            res_xl_decode = res_xl_decode * 2 - 256

            res_quan = res_xl_decode.permute(2, 0, 1) * one_hots
            mask_tensor = torch.from_numpy(threshold_mask).permute(0, 3, 1, 2).squeeze(0)
            res_quan = torch.where(mask_tensor, torch.tensor(0, dtype=res_quan.dtype), res_quan)


        if one_hots.sum() == 0:
            # reward = 0
            output1 = output
            bpp_xl = 0
        else:
            output1 = output + res_quan.unsqueeze(0)  #
            # saveimg(output1, 'results/' + '.png')
        self.t += 1

        # 更新 observation
        one_hot1 = torch.ones(256, 512)  # （h, w)
        m += 1
        one_hot1[input_lable != m.item()] = 0  # Sm
        sm1 = one_hot1.unsqueeze(0)
        sm1 = torch.cat((sm1, sm1, sm1), 0)  # [3, 256, 512]
        # sm1 = torch.cat((one_hot1, one_hot1, one_hot1), 0)  # [3, 256, 512]
        fm1 = one_hot1 * res
        observation_ = [sm1, fm1, m]        #m+1 observation[sm, fm, m+1]

        return observation_, output1, bpp_xl








