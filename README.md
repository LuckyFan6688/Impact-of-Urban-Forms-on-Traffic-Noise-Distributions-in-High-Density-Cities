# Urban Traffic Noise

Analysis code for *Impact of Urban Forms on Traffic Noise Distributions in High-Density Cities*.

The repository contains analysis code, configuration files, synthetic examples, and selected summary results reported in the manuscript. Original study datasets are not included.

## Methods

| Module | Contents |
| --- | --- |
| `01_Noise_Prediction_Model` | Seven-channel inputs; EfficientNet-B4 encoder, U-Net++ decoder and CBAM; architecture and inference example. |
| `02_Indicator_Calculation` | Nineteen urban-form indicators, spatial noise statistics and attenuation intervals. |
| `03_Moran_Clustering` | Queen spatial weights, Global and Local Moran's I, and two-stage K-means with SSE and silhouette diagnostics. |
| `04_XGBoost_SHAP` | Regression and multiclass classification, random and spatial cross-validation, buffered validation, TreeExplainer and cross-fold importance stability. |

The workflow is noise prediction → indicator calculation → spatial statistics and typology → XGBoost and SHAP. Modules 03 and 04 import the shared indicator functions from Module 02; retain the directory structure when running them.

## Installation

Use Python 3.10 or later in a virtual environment:

```bash
python -m venv .venv
# Activate .venv using the command appropriate to your operating system.
python -m pip install -r requirements.txt
```

The examples run on the CPU without downloading pretrained weights. They do not require CadnaA or access to study data.

## Run the synthetic examples

From the repository root:

```bash
python 01_Noise_Prediction_Model/example_inference.py --output outputs/noise_prediction
python 02_Indicator_Calculation/example.py --output outputs/indicators
python 03_Moran_Clustering/example.py --output outputs/moran_clustering
python 04_XGBoost_SHAP/example.py --output outputs/xgboost_shap
```

All example inputs are artificial. Example outputs test the software workflow and are not empirical validation results from the manuscript. In Module 01, randomly initialized weights produce a normalized network response rather than calibrated sound levels.

The XGBoost example uses 12 trees and depth 3 for a short software test. Add `--full-parameters` to use the recorded 400/700/800-tree configurations for noise regression, attenuation regression and classification, respectively.

## Inputs, outputs and parameters

**Noise prediction.** `example_data/synthetic_rgb.npy` has shape `[3, 256, 256]`; `synthetic_auxiliary.npy` has four aligned channels in building-height, road-distance, road-mask and road-width order. RGB values are divided by 255. The model receives `[batch, 7, height, width]` and returns one sigmoid output channel. `model_config.yaml` records architecture, channel order and training settings. The example exports an array, a figure and a small JSON summary.

**Indicator calculation.** `urban_form_indicators.py` accepts aligned building-height, road-mask, road-width and land-class arrays. `indicator_definitions.csv` lists the 19 output fields, abbreviations, formulas, units and required inputs. The artificial five-channel example additionally includes noise pixels. Spatial `Ln` is `percentile(pixels, 100-n)`, using linear percentile interpolation, and describes variation across pixels within an analysis unit. `Lave` is the arithmetic mean of those pixel levels. These are spatial statistics, not time-exceedance measurements at a fixed receiver. Attenuation targets use absolute extrema: `Delta_H = Lmax - L(n_HM)`, `Delta_M = L(n_HM) - L(n_ML)`, and `Delta_L = L(n_ML) - Lmin`, with full-precision exceedance boundaries defined in `attenuation_intervals.py`.

**Moran's I and clustering.** The example uses a fictional dimensionless grid and artificial pixel arrays. `clustering_config.yaml` records Queen adjacency, reciprocal nearest-neighbour repair for islands, row standardization, 999 permutations, and the significance threshold. Interval discovery uses standardized spatial quantiles and an explicit preferred/minimum count of three. Its record range is Q90−Q10; centre ordering uses Q95−Q10. The example demonstrates interval discovery on one artificial input set and four-type classification using the recorded boundaries on a second artificial grid. The second stage fixes four types; SSE and silhouette values are diagnostics rather than an unconstrained selection of the four-type count. Outputs include synthetic global/local statistics, type summaries and diagnostic figures.

**XGBoost and SHAP.** The CSV inputs use the 19 predictor fields listed in `model_config.yaml`, noise-response columns and analysis-unit centres. The included coordinates describe fictional locations. For external study inputs, use verified metre coordinates in EPSG:32650. `spatial_cross_validation.py` builds 2,000 m blocks, applies five-fold GroupKFold, and excludes training centres at distances strictly below 1,000 m from any test centre during buffered validation. Each coordinate of the fixed block origin is the city minimum rounded down to the nearest 1,000 m. Analysis units and validation blocks are distinct. Random folds use seed 42. Imputation is fitted within each training fold. Regression reports pooled out-of-fold R², and RMSE as well as fold metrics; classification reports classification metrics. Held-out SHAP importance is summarized by mean absolute values, pairwise Spearman rank correlation and Top-5 overlap. SHAP associations do not establish causality. The example covers seven responses and one four-class task for each of three artificial datasets.

## Selected manuscript results

The following CSV files transcribe reported manuscript summaries; running synthetic examples does not recalculate them:

| File | Source and scope |
| --- | --- |
| `01_Noise_Prediction_Model/results/overall_accuracy.csv` | Section 2.2; reported model-versus-simulation error bounds and separate Tianjin simulation-versus-field validation. Inequality operators are retained. |
| `03_Moran_Clustering/results/appendix_table_3.csv` | Appendix Table 3; city-level Global Moran statistics. |
| `03_Moran_Clustering/results/clustering_summary.csv` | Explicitly reported city/type percentages; a partial summary rather than a complete type distribution. |
| `04_XGBoost_SHAP/results/appendix_table_4.csv` | Appendix Table 4; 21 regression models and their disclosed validation/stability summaries. |
| `04_XGBoost_SHAP/results/shap_reported_predictors.csv` | Section 3.3.1; qualitative predictor lists and ordering explicitly reported in the text. |

## Citation and licence

When referring to this work, cite *Impact of Urban Forms on Traffic Noise Distributions in High-Density Cities* . 

See `LICENSE` for terms. Third-party dependencies retain their own licences.
