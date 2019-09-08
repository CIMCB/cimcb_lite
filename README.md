<img src="cimcb_logo.png" alt="drawing" width="400"/>

# cimcb lite 
cimcb_lite is a lite version of the cimcb package containing a small number of basic tools for the statistical analysis of untargeted and targeted metabolomics data.

## Installation

### Dependencies
cimcb_lite requires:
- Python (>=3.5)
- Bokeh (>=1.0.0)
- NumPy
- SciPy
- scikit-learn
- Statsmodels
- tqdm

### User installation
The recommend way to install cimcb_lite and dependencies is to using ``conda``:
```console
conda install -c cimcb cimcb_lite
```
or ``pip``:
```console
pip install cimcb_lite
```
Alternatively, to install directly from github:
```console
pip install https://github.com/cimcb/cimcb_lite/archive/master.zip
```

### API
For futher detail on the usage refer to the docstring.

#### cimcb_lite.model
- [PLS_SIMPLS](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/model/PLS_SIMPLS.py#L14-L36): Partial least-squares regression using the SIMPLS algorithm.
  - [train](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/model/PLS_SIMPLS.py#L43-L58): Fit the PLS model, save additional stats (as attributes) and return Y predicted values.
  - [test](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/model/PLS_SIMPLS.py#L105-L117): Calculate and return Y predicted value.
  - [evaluate](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/model/BaseModel.py#L40-L56): Plots a figure containing a Violin plot, Distribution plot, ROC plot and Binary Metrics statistics.
  - [calc_bootci](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/model/BaseModel.py#L191-L201): Calculates bootstrap confidence intervals based on bootlist.
  - [plot_featureimportance](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/model/BaseModel.py#L211-L212): Plots feature importance metrics.
  - [plot_permutation_test](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/model/BaseModel.py#L253-L254): Plots permutation test figures.

#### cimcb_lite.plot
- [boxplot](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/plot/boxplot.py#L8-L18): Creates a boxplot using Bokeh.
- [distribution](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/plot/distribution.py#L6-L16): Creates a distribution plot using Bokeh.
- [pca](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/plot/pca.py#L10-L17): Creates a PCA scores and loadings plot using Bokeh.
- [permutation_test](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/plot/permutation_test.py#L13-L27): Creates permutation test plots using Bokeh.
- [roc_plot](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/plot/roc.py#L11-L24): Creates a rocplot using Bokeh.
- [scatter](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/plot/scatter.py#L6-L16): Creates a scatterplot using Bokeh.
- [scatterCI](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/plot/scatterCI.py#L7-L14): Creates a scatterCI plot using Bokeh.

#### cimcb_lite.cross_val
- [kfold](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/cross_val/kfold.py#L14-L42): Exhaustitive search over param_dict calculating binary metrics.

#### cimcb_lite.bootstrap
- [Perc](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/bootstrap/Perc.py#L6-L35): Returns bootstrap confidence intervals using the percentile boostrap interval.
- [BC](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/bootstrap/BC.py#L8-L37): Returns bootstrap confidence intervals using the bias-corrected boostrap interval.
- [BCA](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/bootstrap/BCA.py#L8-L36): Returns bootstrap confidence intervals using the bias-corrected and accelerated boostrap interval.

#### cimcb_lite.utils
- [binary_metrics](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/utils/binary_metrics.py#L5-L23): Return a dict of binary stats with the following metrics: R2, auc, accuracy, precision, sensitivity, specificity, and F1 score.
- [ci95_ellipse](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/utils/ci95_ellipse.py#L6-L28): Construct a 95% confidence ellipse using PCA.
- [knnimpute](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/utils/knnimpute.py#L7-L22): kNN missing value imputation using Euclidean distance.
- [load_dataXL](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/utils/load_dataXL.py#L7-L29): Loads and validates the DataFile and PeakFile from an excel file.
- [nested_getattr](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/utils/nested_getattr.py#L4-L5): getattr for nested attributes.
- [scale](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/utils/scale.py#L4-L42): Scales x (which can include nans) with method: 'auto', 'pareto', 'vast', or 'level'.
- [table_check](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/utils/table_check.py#L4-L17): Error checking for DataTable and PeakTable (used in load_dataXL).
- [univariate_2class](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/utils/univariate_2class.py#L8-L35): Creates a table of univariate statistics (2 class).
- [wmean](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/utils/wmean.py#L4-L19): Returns Weighted Mean. Ignores NaNs and handles infinite weights.

### License
cimcb_lite is licensed under the MIT license. 

### Authors
- [Kevin Mendez](https://github.com/KevinMMendez/)
- [David Broadhurst](https://scholar.google.ca/citations?user=M3_zZwUAAAAJ&hl=en)

### Correspondence
Professor David Broadhurst, Director of the Centre for Integrative Metabolomics & Computation Biology at Edith Cowan University.
