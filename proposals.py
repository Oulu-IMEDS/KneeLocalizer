import numpy as np
import pydicom as dicom


def read_dicom(filename, cut_min=5, cut_max=99):
    """Read DICOM file and convert it to a decent quality uint8 image.

    Parameters
    ----------
    filename: str
        Existing DICOM file filename.
    cut_min: int
        Lowest percentile which is used to cut the image histogram.
    cut_max: int
        Highest percentile.
    """
    try:
        data = dicom.read_file(filename)
        img = np.frombuffer(data.PixelData, dtype=np.uint16).copy().astype(np.float64)

        if data.PhotometricInterpretation == 'MONOCHROME1':
            img = img.max() - img

        lim1, lim2 = np.percentile(img, [cut_min, cut_max])

        img[img < lim1] = lim1
        img[img > lim2] = lim2

        img -= lim1

        img /= img.max()
        img *= 255
        img = img.astype(np.uint8)

        img = img.reshape((data.Rows, data.Columns))
        return img, data.ImagerPixelSpacing[0]
    except:
        return None


def get_joint_y_proposals(img, av_points=11, margin=0.25):
    """Return Y-coordinates of the joint approximate locations."""

    R, C = img.shape

    # Sum the middle if the leg is along the X-axis
    segm_line = np.sum(img[int(R * margin):int(R * (1 - margin)),
                           int(C / 3):int(C - C / 3)], axis=1)
    # Smooth segmentation line and find the absolute of the derivative
    segm_line = np.abs(np.convolve(
        np.diff(segm_line), np.ones((av_points, )) / av_points)[(av_points-1):])

    # Get top tau % of the peaks
    peaks = np.argsort(segm_line)[::-1][:int(0.1 * R * (1 - 2 * margin))]
    return peaks[::10] + int(R * margin)
