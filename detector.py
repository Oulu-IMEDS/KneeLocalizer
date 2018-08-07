from __future__ import print_function
import pydicom as dicom
import numpy as np
import cv2
import argparse
import os
import time
from proposals import read_dicom, get_joint_y_proposals

# This can be uncommented in case if parallel processing is needed
from joblib import Parallel, delayed



def worker(i):
    """
    This function can be applied to list of files in fnames, which is a global array
    """
    res_read = read_dicom(os.path.join(DIR, fnames[i]))
    if res_read is None:
        print("failed on {0}, {1}".format(i, fnames[i]))
        return ' '.join(map(str,[fnames[i]] + [-1]*4 +[-1]*4))

    img, spacing = res_read
    R, C = img.shape
    split_point = C/2

    right_l = img[:,:split_point]
    left_l = img[:,split_point:]

    prop = get_joint_y_proposals(right_l)

    # We will store the coordinates of the top left and the bottom right corners of the bounding box
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)


    # Making proposals for the right leg
    R, C = right_l.shape
    displacements = range(-C//4,1*C//4+1,step)
    best_score = -999999999
    sizepx = int(sizemm/spacing) # Proposal size

    for y_coord in prop:
        for x_displ in displacements:
            for scale in scales:
                if C/2+x_displ-R/scale/2 >= 0:
                    # Candidate ROI
                    roi = np.array([C/2+x_displ-R/scale/2, y_coord-R/scale/2, R/scale, R/scale]).astype(int)
                    x1, y1 = roi[0], roi[1]
                    x2, y2 = roi[0]+roi[2], roi[1]+roi[3]
                    patch = cv2.resize(img[y1:y2,x1:x2],(64, 64))

                    hog_descr = hog.compute(patch,winStride,padding)
                    score = np.inner(w,hog_descr.ravel())+b

                    if score > best_score:
                        jc = np.array([C/2+x_displ, y_coord])
                        best_score = score


    roi_R = np.array([jc[0]-sizepx//2, jc[1]-sizepx//2, jc[0]+sizepx//2, jc[1]+sizepx//2])
    # Making proposals for the left leg
    R, C = left_l.shape
    displacements = range(-C//4,1*C//4+1,step)
    prop = get_joint_y_proposals(left_l)
    best_score = -999999999
    for y_coord in prop:
        for x_displ in displacements:
            for scale in scales:
                if split_point+x_displ+R/scale/2 < img.shape[1]:
                    roi = np.array([split_point+C/2+x_displ-R/scale/2, y_coord-R/scale/2, R/scale, R/scale]).astype(int)

                    x1, y1 = roi[0], roi[1]
                    x2, y2 = roi[0]+roi[2], roi[1]+roi[3]
                    patch = np.fliplr(cv2.resize(img[y1:y2,x1:x2],(64, 64)))

                    hog_descr = hog.compute(patch,winStride,padding)
                    score = np.inner(w,hog_descr.ravel())+b

                    if score > best_score:
                        jc = np.array([split_point+C/2+x_displ, y_coord])
                        best_score = score

    roi_L = np.array([jc[0]-sizepx//2, jc[1]-sizepx//2, jc[0]+sizepx//2, jc[1]+sizepx//2])

    print("Done with {}, {}".format(i, fnames[i]))
    return ' '.join(map(str,[fnames[i]] + np.round(roi_L).astype(int).tolist() + np.round(roi_R).astype(int).tolist()))




if __name__ == "__main__":

    start = time.time()
    global DIR, w, b, fnames, winSize
    global blockSize, blockStride, cellSize, winStride
    global padding, nbins, step, scales, sizemm
    sizemm = 120
    winSize = (64,64)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    winStride = (64,64)
    padding = (0,0)
    nbins = 9
    scales = [ 3.2, 3.3, 3.4, 3.6, 3.8]
    step=95

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir")
    args = parser.parse_args()

    DIR = os.path.abspath(args.dir)
    fnames = os.listdir(DIR)
    w, b = np.load('svm_model.npy')

    res = Parallel(n_jobs=6)(delayed(worker)(i) for i in range(len(fnames)))
    with open('detection_results.txt','w') as f:
        for entry in res:
            f.write(entry+'\n')

    print('Script execution took', time.time()-start, 'seconds.')
