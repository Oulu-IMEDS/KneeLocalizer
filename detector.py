import os
import time
import argparse
from multiprocessing import Pool, cpu_count

import numpy as np
import cv2
from tqdm import tqdm

from proposals import read_dicom, get_joint_y_proposals


def worker(fname, path_input, size_mm, win_size, win_stride,
           block_size, block_stride, cell_size, padding, nbins,
           scales, step, svm_w, svm_b):
    res_read = read_dicom(os.path.join(path_input, fname))
    if res_read is None:
        ret = [fname, ] + [-1, ] * 4 + [-1, ] * 4
        return ' '.join([str(e) for e in ret])

    img, spacing = res_read
    R, C = img.shape
    split_point = C // 2

    right_leg = img[:, :split_point]
    left_leg = img[:, split_point:]

    sizepx = int(size_mm / spacing)  # Proposal size

    # We will store the coordinates of the top left and
    # the bottom right corners of the bounding box
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

    # Make proposals for the right leg
    R, C = right_leg.shape
    displacements = range(-C // 4, 1 * C // 4 + 1, step)
    prop = get_joint_y_proposals(right_leg)
    best_score = -np.inf

    for y_coord in prop:
        for x_displ in displacements:
            for scale in scales:
                if C/2 + x_displ - R / scale / 2 >= 0:
                    # Candidate ROI
                    roi = np.array([C / 2 + x_displ - R / scale / 2,
                                    y_coord - R / scale / 2,
                                    R / scale, R / scale], dtype=np.int)
                    x1, y1 = roi[0], roi[1]
                    x2, y2 = roi[0] + roi[2], roi[1] + roi[3]
                    patch = cv2.resize(img[y1:y2, x1:x2], (64, 64))

                    hog_descr = hog.compute(patch, win_stride, padding)
                    score = np.inner(svm_w, hog_descr.ravel()) + svm_b

                    if score > best_score:
                        jc = np.array([C / 2 + x_displ, y_coord])
                        best_score = score

    roi_R = np.array([jc[0] - sizepx // 2,
                      jc[1] - sizepx // 2,
                      jc[0] + sizepx // 2,
                      jc[1] + sizepx // 2]).round().astype(np.int)

    # Make proposals for the left leg
    R, C = left_leg.shape
    displacements = range(-C // 4, 1 * C // 4 + 1, step)
    prop = get_joint_y_proposals(left_leg)
    best_score = -np.inf

    for y_coord in prop:
        for x_displ in displacements:
            for scale in scales:
                if split_point + x_displ + R / scale / 2 < img.shape[1]:
                    roi = np.array([split_point + C / 2 + x_displ - R / scale / 2,
                                    y_coord - R / scale / 2,
                                    R / scale, R / scale], dtype=np.int)
                    x1, y1 = roi[0], roi[1]
                    x2, y2 = roi[0] + roi[2], roi[1] + roi[3]
                    patch = np.fliplr(cv2.resize(img[y1:y2, x1:x2], (64, 64)))

                    hog_descr = hog.compute(patch, win_stride, padding)
                    score = np.inner(svm_w, hog_descr.ravel()) + svm_b

                    if score > best_score:
                        jc = np.array([split_point + C / 2 + x_displ, y_coord])
                        best_score = score

    roi_L = np.array([jc[0] - sizepx // 2,
                      jc[1] - sizepx // 2,
                      jc[0] + sizepx // 2,
                      jc[1] + sizepx // 2]).round().astype(np.int)

    return ' '.join(map(str, [fname, ] + roi_L.tolist() + roi_R.tolist()))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_input', "--dir")
    parser.add_argument('--fname_output', "--output",
                        default='../detection_results.txt')

    args = parser.parse_args()
    args.path_input = os.path.abspath(args.path_input)
    args.fname_output = os.path.abspath(args.fname_output)
    return args


if __name__ == "__main__":
    args = parse_args()

    ts_start = time.time()

    size_mm = 120
    win_size = (64, 64)
    win_stride = (64, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    padding = (0, 0)
    nbins = 9
    scales = [3.2, 3.3, 3.4, 3.6, 3.8]
    step = 95

    svm_w, svm_b = np.load('svm_model.npy', encoding='bytes')
    
    def worker_partial(fname):
        return worker(fname, args.path_input, size_mm, win_size, win_stride,
                      block_size, block_stride, cell_size, padding, nbins,
                      scales, step, svm_w, svm_b)

    fnames = os.listdir(args.path_input)
    
    with Pool(cpu_count()) as pool:
        res = list(tqdm(pool.imap(
            worker_partial, iter(fnames)), total=len(fnames)))
        
    with open(args.fname_output, 'w') as f:
        for entry in res:
            f.write(entry + '\n')

    ts_end = time.time() - ts_start
    print('Script execution took {} seconds'.format(ts_end))
