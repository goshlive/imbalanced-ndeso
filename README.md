# NDESO
**Neighbor Displacement-based Enhanced Synthetic Oversampling (NDESO)** is a resampling method used to handle sparse and imbalanced multiclass data with significantly overlapping data points. This method uses _k_-neighbor-based displacement to adjust noisy data points by moving them closer to their center, which is further enhanced using random oversampling techniques to balance the dataset.

The **_smotecdnn.py_** file is the original version of the SMOTE-CDNN sampler, which can also be found at [https://github.com/coksvictoria/SMOTE-CDNN](https://github.com/coksvictoria/SMOTE-CDNN). This version was published by the authors in their paper, accessible through the DOI: [https://doi.org/10.1016/j.asoc.2023.110895](https://doi.org/10.1016/j.asoc.2023.110895). We use this code to reference the sampler during comparative testing with our proposed method.

The **_data_** folder includes multiple public datasets featuring multiclass imbalanced data sourced from Knowledge Extraction (https://sci2s.ugr.es/keel/imbalanced.php) and OpenML (https://www.openml.org/search?type=data).

The **_output_** folder holds the results of our comparison of NDESO with various other resampling techniques across different classifiers and datasets.

For instructions on how to perform the testing, refer to the **TEST.ipynb** file, which provides an example of its usage.

The experimental results are detailed in a paper, with the preprint version accessible at: [https://arxiv.org/abs/2501.04099].
