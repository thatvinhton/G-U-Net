import numpy as np
from scipy.ndimage.morphology import binary_dilation, binary_opening
from scipy.ndimage import maximum_filter
import cv2


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


def nnz(mask):
    return np.sum((mask > 1e-9).astype(np.int32))


def calAJI(predicted_map, label):

    gt_list = np.unique(label)
    gt_list = gt_list[1:]
    ngt = len(gt_list)

    predicted_map = predicted_map.astype(np.float32)
    pr_list = np.unique(predicted_map)
    pr_list = pr_list[1:]
    npredicted = len(pr_list)
    mark_pr = np.zeros(npredicted)

    overall_correct_count = 0
    union_pixel_count = 0

    for i in range(ngt):

        gt = (label == gt_list[i]).astype(np.int32)

        predicted_match = np.multiply(gt, predicted_map)

        print(nnz(predicted_match), ' ', nnz(gt))

        if nnz(predicted_match) == 0:
            union_pixel_count += nnz(gt)
        else:
            predicted_nuc_index = np.unique(predicted_match)
            predicted_nuc_index = predicted_nuc_index[1:]

            JI = 0
            for j in range(len(predicted_nuc_index)):
                matched = (predicted_map == predicted_nuc_index[j]).astype(np.float32)
                nJI = nnz(np.logical_and(matched>0, gt>0).astype(np.int32)) / nnz(np.logical_or(matched>0, gt>0).astype(np.int32))

                if nJI > JI:
                    best_match = predicted_nuc_index[j]
                    JI = nJI

            predicted_nuclei = (predicted_map == best_match).astype(np.int32)

            union_pixel_count += nnz(np.logical_or(gt > 0, predicted_nuclei > 0).astype(np.int32))
            overall_correct_count += nnz(np.logical_and(gt > 0, predicted_nuclei > 0).astype(np.int32))

            index = 0
            for j in range(len(pr_list)):
                if pr_list[j] == best_match:
                    index = j
                    break
            # index = pr_list.index(best_match)
            mark_pr[index] += 1

    for i in range(npredicted):
        if mark_pr[i] == 0:
            unused_nuclei = (predicted_map == pr_list[i]).astype(np.int32)
            union_pixel_count += nnz(unused_nuclei)

    if union_pixel_count == 0:
        return 1

    return overall_correct_count / union_pixel_count




