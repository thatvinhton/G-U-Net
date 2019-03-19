import os
from PIL import Image
import numpy as np
import scipy.io as sio
from scipy.ndimage.morphology import binary_dilation, binary_opening
from scipy.ndimage import maximum_filter
import cv2
import matplotlib.pyplot as plt


OPENING_FULL_MASK_ITER = 1
OPENING_CELL_SEED_ITER = 2
THRES_SEED = 0.85
FINAL_EXPAND_STEP = 2

VISUALIZE = 1


def get_random_color():
    return [255 * np.random.rand(), 255 * np.random.rand(), 255 * np.random.rand()]


def on_boundary(id, i, j):
    if id[i, j] == 0:
        return False

    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            x = i + di
            y = j + dj

            if x in range(id.shape[0]) and y in range(id.shape[1]):
                if id[i, j] != id[x, y]:
                    return True


def visualize(img):
    res = np.zeros(shape=(img.shape[0], img.shape[1], 3))

    num_cells = np.max(img)

    for i in range(num_cells + 1):
        if i == 0:
            continue

        color = get_random_color()

        mask = (img == i).astype(np.uint8)
        res[:, :, 0] = res[:, :, 0] + mask * color[0]
        res[:, :, 1] = res[:, :, 1] + mask * color[1]
        res[:, :, 2] = res[:, :, 2] + mask * color[2]

    return res


def expand_region_one_step(limit_region, cur_id):
    label = (cur_id > 0).astype(np.uint8)

    label = binary_dilation(label) - label

    true_expanding = np.logical_and(label > 0, limit_region > 0).astype(np.uint8)

    temp_id = maximum_filter(cur_id, 3)

    temp_id = true_expanding * temp_id

    new_id = cur_id + temp_id

    return new_id


def post_process(img, opening_full_mask_iter, opening_cell_seed_iter, thres, final_expanding_iter):
    h, w = img.shape[0], img.shape[1]

    label = np.zeros([h, w])

    full_mask = (np.logical_and(img[:, :, 2] > img[:, :, 0], img[:, :, 2] > img[:, :, 1]))
    cell_seed = (img[:, :, 2] > thres)

    # opening the full mask
    if opening_full_mask_iter > 0:
        full_mask_opened = binary_opening(full_mask, iterations=opening_full_mask_iter).astype(np.uint8)
    else:
        full_mask_opened = full_mask.astype(np.uint8)

    # opening the cell seed mask
    if opening_cell_seed_iter > 0:
        cell_seed_opened = binary_opening(cell_seed, iterations=opening_cell_seed_iter).astype(np.uint8)
    else:
        cell_seed_opened = cell_seed.astype(np.uint8)

    # generate distinct id for every region
    _, cell_id = cv2.connectedComponents(cell_seed_opened.astype(np.uint8), connectivity=4)

    # expanding method
    while True:
        new_cell_id = expand_region_one_step(full_mask_opened, cell_id)

        if np.sum(cell_id - new_cell_id) == 0:
            break

        cell_id = new_cell_id

    # assign id for missing region
    left_region = (full_mask_opened - (cell_id > 0).astype(np.uint8))
    _, left_cell_id = cv2.connectedComponents(left_region, connectivity=4)

    cur_id = np.max(cell_id)
    left_cell_id[left_cell_id > 0] += cur_id

    cell_id = cell_id + left_cell_id

    # apply final transformation

    for i in range(final_expanding_iter):
        cell_id = expand_region_one_step(np.ones(shape=(h, w)).astype(np.uint8), cell_id)

    return cell_id


if __name__ == '__main__':

    imgRoot = '/home/cig/Desktop/result_fromUnet/0809_unet_ext2_TTA_aug/result_UNet_figure7'
    imgResDir = '/home/cig/Desktop/post_processing_result_UNet_figure7'

    if not os.path.exists(os.path.join(imgResDir, 'mat')):
        os.makedirs(os.path.join(imgResDir, 'mat'))

    if VISUALIZE != 0:
        if not os.path.exists(os.path.join(imgResDir, 'img')):
            os.makedirs(os.path.join(imgResDir, 'img'))

    file_list = sorted([f for f in os.listdir(imgRoot)])

    for img_full_name in file_list:
        img_name = img_full_name[:-4]

        print(img_name)

        img = np.array(Image.open(os.path.join(imgRoot, img_full_name)))
        img = img / 255.0

        predicted_map = post_process(img, OPENING_FULL_MASK_ITER, OPENING_CELL_SEED_ITER, THRES_SEED, FINAL_EXPAND_STEP)

        sio.savemat(os.path.join(imgResDir, 'mat', img_name + '.mat'), mdict={'predicted_map': predicted_map})

        if VISUALIZE != 0:
            visualize_img = visualize(predicted_map)

            img_res = Image.fromarray((visualize_img).astype(np.uint8))
            img_res.save(os.path.join(imgResDir, 'img', img_name + '.png'))
