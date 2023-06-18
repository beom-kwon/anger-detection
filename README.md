# Contents
1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [How to run](#how-to-run)
4. [Citation](#citation)
5. [License](#license)

# Introduction

On this web page, we provide the Python implementation of the anger detection method proposed in our paper titled '[Multi-Time Window Feature Extraction Technique for Anger Detection in Gait Data](https://doi.org/10.9708/jksci.2023.28.04.041).'

# Requirements

The proposed method was implemented using Python 3.9.16. To implement five machine learning algorithms (logistic regression, k-nearest neighbors, Naive Bayes, c-suport vector machine, and decision tree), we utilized scikit-learn 1.2.1. Additionally, we employed matplotlib 3.6.2 and numpy 1.23.5.

# How to Run

## 1. Dataset Preparation

* We utilized the Body Movement Library (BML) dataset, which includes gait sequences from 30 subjects. The dataset utilizes two file formats, namely CSM and PTD. In our study, we specifically used PTD file format.
* If you wish to download this dataset, please click [here](https://paco.psy.gla.ac.uk/?page_id=14973) to access the download link.

## 2. Multi-Time Window Feature Extraction

Please execute 'feature_extraction.py.' After running the .py file, you will obtain an npz file containing feature vectors.

## 3. Ensemble Learning

Please execute 'ensemble_learning.py.' After running the .py file, you will see the performance evaluation results of our proposed ensemble model.

# Citation

Please cite this paper in your publications if it helps your research.

```
@article{kwon2023multi,
  author={Kwon, Beom and Oh, Taegeun},
  journal={Journal of The Korea Society of Computer and Information},
  title={Multi-Time Window Feature Extraction Technique for Anger Detection in Gait Data},
  year={2023},
  volume={28},
  number={4}
  pages={41-51},
  doi={10.9708/jksci.2023.28.04.041}
}
```
Paper link:
* [Multi-Time Window Feature Extraction Technique for Anger Detection in Gait Data](https://doi.org/10.9708/jksci.2023.28.04.041)

# License

Our codes are freely available for non-commercial use.
