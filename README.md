# NDESO
**Neighbor Displacement-based Enhanced Synthetic Oversampling (NDESO)** is a resampling method used to handle sparse and imbalanced multiclass data with significantly overlapping data points. This method uses k-neighbor-based displacement to adjust noisy data points by moving them closer to their center, which is further enhanced using random oversampling techniques to balance the dataset.

The **_data_** folder includes multiple public datasets featuring multiclass imbalanced data sourced from Knowledge Extraction (https://sci2s.ugr.es/keel/imbalanced.php) and OpenML (https://www.openml.org/search?type=data).

The **_output_** folder holds the results of our comparison of NDESO with various other resampling techniques across different classifiers and datasets.

For instructions on how to perform the testing, refer to the **TEST.ipynb** file, which provides an example of its usage.
