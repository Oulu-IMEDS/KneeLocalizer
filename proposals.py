import dicom
import scipy.io as spio
import numpy as np


def read_dicom(filename, cut_min=5, cut_max=99):
    """
    The fucntion tries to read the dicom file and convert it to a decent quality uint8 image

    gamma_A - scaling factor, which is needed to make the values on the image to be > 1
    gamma_B - gamma correction degree
    cut_min - lowest percentile which is used to cut the image histogram
    cut_max - highest percentile

    """
    try:
        data = dicom.read_file(filename)
        img = np.frombuffer(data.PixelData,dtype=np.uint16).copy().astype(np.float64)

        if data.PhotometricInterpretation == 'MONOCHROME1':
            img = img.max()-img

        lim1, lim2 = np.percentile(img, [cut_min, cut_max])

        img[img < lim1] = lim1
        img[img > lim2] = lim2

        img -= lim1

        img /= img.max()
        img *= 255
        img = img.astype(np.uint8)

        img = img.reshape((data.Rows,data.Columns))
        return img, data.ImagerPixelSpacing[0]
    except:
        return None


def get_joint_y_proposals(img, av_points=11, margin=0.25):
    """
    Returns Y-coordinates of the aproximate Joint locations
    """
    R, C = img.shape
    # Summing the middle if the leg along the X-axis

    segm_line = np.sum(img[int(R*margin):int(R*(1-margin)),C/3:C-C/3],1)
    # Making segmentation line smooth and finding the absolute of the derivative
    segm_line = np.abs(np.convolve(np.diff(segm_line), np.ones((av_points,))/av_points)[(av_points-1):])

    # Getting top tau % of the peaks
    peaks = np.argsort(segm_line)[::-1][:int(0.1*R*(1-2*margin))]
    return peaks[::10]+int(R*margin)
