import numpy as np
import math

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


def Accu_MIoU(end_gray, real_gray):
    gray_end_pixel_values = list(np.unique(end_gray))
    print('=========================================')
    # print(gray_end_pixel_values)
    gray_real_pixel_values = list(np.unique(real_gray))
    # print(gray_real_pixel_values)

    num_classes = list(set(gray_end_pixel_values + gray_real_pixel_values))
    # print(num_classes)
    pre_right = list(set(gray_end_pixel_values) & set(gray_real_pixel_values))
    # print(pre_right)
    Accu = len(pre_right) / len(num_classes)
    print(f'Accu_Syn: {Accu:.4f}')
    ious, _ = calculate_iou(real_gray, end_gray, pre_right, print_c_i=True)
    miou = sum(ious)/len(pre_right)
    print(f'MIoU_Syn: {miou:.4f}')
    print('=========================================')
    return Accu, miou