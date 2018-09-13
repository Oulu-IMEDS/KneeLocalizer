import os
import time
import argparse
from multiprocessing import Pool, cpu_count

import numpy as np
import cv2
from tqdm import tqdm

from proposals import read_dicom, get_joint_y_proposals, preprocess_xray


class KneeLocalizer:
    def __init__(self, svm_model_path='svm_model.npy', size_mm=120):
        super().__init__()
        self.size_mm = 120
        self.win_size = (64, 64)
        self.win_stride = (64, 64)
        self.block_size = (16, 16)
        self.block_stride = (8, 8)
        self.cell_size = (8, 8)
        self.padding = (0, 0)
        self.nbins = 9
        self.scales = [3.2, 3.3, 3.4, 3.6, 3.8]
        self.step = 95
        self.size_mm = size_mm
        self.svm_w, self.svm_b = np.load(svm_model_path, encoding='bytes')

    def predict(self, fname, spacing=None):
        """Localizes the left and the right knee joints on PA X-ray.

        :param fname: str or numpy.array
            Filename of the DICOM imahe, or already extracted uint16 np array
        :param spacing: float or None
            Spacing extracted from teh previously read DICOM
        :return: list
            List of lists. The first list has the bbox for the left knee joint. The second list has the bbox
            for the right knee joint.
        """

        size_mm = self.size_mm
        win_size = self.win_size
        win_stride = self.win_stride
        block_size = self.block_size
        block_stride = self.block_stride
        cell_size = self.cell_size
        padding = self.padding
        nbins = self.nbins
        scales = self.scales
        step = self.step
        size_mm = self.size_mm
        svm_w = self.svm_w
        svm_b = self.svm_b

        if isinstance(fname, str):
            res_read = read_dicom(fname)
            if res_read is None:
                return None
            if len(res_read) != 2:
                return None
            img, spacing = res_read
            img = preprocess_xray(img)
        elif isinstance(fname, np.ndarray):
            img = fname
            if spacing is None:
                raise ValueError
        else:
            raise ValueError

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
                    if C / 2 + x_displ - R / scale / 2 >= 0:
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

        return [roi_L.tolist(), roi_R.tolist()]


def worker(fname, path_input, localizer):

    res_read = read_dicom(os.path.join(path_input, fname))
    if res_read is None:
        ret = [fname, ] + [-1, ] * 4 + [-1, ] * 4
        return ' '.join([str(e) for e in ret])

    img, spacing = res_read
    img = preprocess_xray(img)
    try:
        res = localizer.predict(img, spacing)
    except:
        res = [[-1]*4, [-1]*4]

    if res is None:
        res = [[-1]*4, [-1]*4]
    return ' '.join(map(str, [fname, ] + res[0] + res[1]))


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

    localizer = KneeLocalizer()

    def worker_partial(fname):
        return worker(fname, args.path_input, localizer)

    fnames = os.listdir(args.path_input)
    
    with Pool(cpu_count()) as pool:
        res = list(tqdm(pool.imap(
            worker_partial, iter(fnames)), total=len(fnames)))
        
    with open(args.fname_output, 'w') as f:
        for entry in res:
            f.write(entry + '\n')

    ts_end = time.time() - ts_start
    print('Script execution took {} seconds'.format(ts_end))
