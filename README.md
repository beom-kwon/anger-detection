# Contents
1. [Introduction](Introduction)
2. [Requirements](Requirements)
3. [How to run](How-to-run)
4. [Citation](Citation)
5. [License](License)

# Introduction

In this web-page, we provide the Python implementation of the anger detection method proposed in our paper "[Multi-Time Window Feature Extraction Technique for Anger Detection in Gait Data](https://doi.org/10.9708/jksci.2023.28.04.041)." 

# Requirements

The proposed method was implemented using Python 3.9.16. To implement five machine learning algorithms (i.e., logistic regression, k-nearest neighbors, Naive Bayes, c-suport vector machine, and decision tree), we used scikit-learn 1.2.1. In addition, we also used matplotlib 3.6.2. and numpy 1.23.5.

# How to run

## 1. Dataset Preparation

* We used the Body Movement Library (BML) dataset. This dataset contains gait sequences from 30 subjects. Two file formats, called CSM and PTD, are used in the dataset. In our study, we used PTD file format.
* If you want to download this dataset, please click [here](https://paco.psy.gla.ac.uk/?page_id=14973). You can then find the download link.

## 2. Multi-Time Window Feature Extraction

Run "feature_extraction.py." After the py file is executed, you can obtain an npz file that contains feature vectors.

## 3. Ensemble Learning

Run "ensemble_learning.py." After the py file is executed, you can see the performance evaluation results of our proposed ensemble model.

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

Our codes are freely available for free non-commercial use.
