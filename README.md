# Code for the paper from SCIA'17: A novel method for automatic localization of joint area on knee plain radiographs

# Description
Repository contains the code for an automatic knee joint detection on plain radiographs. This can be used to process very large amount of knee x-rays and generate bounding boxes (up to 6 000 000 per day on a high-end computer).

Our package processes the data in a batch mode using multiple threads. To run in on your machine, you have to install the following python dependencies:

* numpy
* pydicom
* opencv-python
* joblib

# How to run
Run the script as follows:
```
python detector.py --dir <dir with DICOM files>
```
# How to cite
If you use our package in your own research, please cite us:

```
@inproceedings{tiulpin2017novel,
  title={A novel method for automatic localization of joint area on knee plain radiographs},
  author={Tiulpin, Aleksei and Thevenot, Jerome and Rahtu, Esa and Saarakkala, Simo},
  booktitle={Scandinavian Conference on Image Analysis},
  pages={290--301},
  year={2017},
  organization={Springer}
}
```
