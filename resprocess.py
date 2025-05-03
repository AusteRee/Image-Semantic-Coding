import numpy as np
import torch


def resProcess(res, gray_label):
    class_gray_label, dict_gray_label, prop = pixel_prop(gray_label, threshold=0.1)
    mask = np.ones_like(gray_label, dtype=np.uint8)
    low_mask = segment_semantic_map(gray_label, block_size=(16, 32), target_values=class_gray_label)
    high_mask = mask - low_mask

    Mask_syn_low = np.expand_dims(low_mask, axis=-1)
    Mask_syn_high = np.expand_dims(high_mask, axis=-1)
    Mask_syn_low = np.repeat(Mask_syn_low, 3, axis=-1)
    Mask_syn_high = np.repeat(Mask_syn_high, 3, axis=-1)

    # threshold
    threshold = 32
    res_min = res.min()
    res_max = res.max()
    # threshold_mv_low = max((-threshold - res_min) * (255 / (res_max - res_min)), 0)
    # threshold_mv_high = min((threshold - res_min) * (255 / (res_max - res_min)), 255)
    threshold_mv_low = max(0 - (threshold/2), 0)
    threshold_mv_high = min((threshold/2), 255)


    # scaled_image = (res + 256) / 2
    # scaled_image = scaled_image.astype(np.uint8)
    scaled_image = res

    # low
    threshold_mask = (Mask_syn_high == 1) & (scaled_image >= threshold_mv_low) & (scaled_image <= threshold_mv_high)

    # zeroing
    scaled_image[threshold_mask] = 0

    scaled_image = (scaled_image + 256) / 2
    scaled_image = scaled_image.astype(np.uint8)

    re = torch.from_numpy(scaled_image)
    return re, threshold_mask


def pixel_prop(gray, threshold=0.1):
    unique_values = list(np.unique(gray))
    unique_values.append(unique_values[-1])

    # percentage
    sum_pixels = gray.size
    pixel_proportion = np.histogram(gray, bins=unique_values)[0] / sum_pixels

    class_sum = []

    dict_gray = dict(zip(unique_values, pixel_proportion))

    # sort
    sorted_dict_gray = dict(sorted(dict_gray.items(), key=lambda item: item[1]))

    cumulative_prop = 0
    for p_v, prop in sorted_dict_gray.items():
        if cumulative_prop <= threshold:
            cumulative_prop += prop
            class_sum.append(p_v)

    # class_sum
    # sorted_dict_gray
    # cumulative_prop
    return class_sum, sorted_dict_gray, cumulative_prop

def segment_semantic_map(semantic_map, block_size=(16, 32), target_values=[]):
    # size
    height, width = semantic_map.shape

    # init
    segmented_mask = np.zeros_like(semantic_map)

    # Segmentation
    for y in range(0, height, block_size[0]):
        for x in range(0, width, block_size[1]):
            block = semantic_map[y:y + block_size[0], x:x + block_size[1]]

            if any(value in block for value in target_values):
                segmented_mask[y:y + block_size[0], x:x + block_size[1]] = 1

    return segmented_mask