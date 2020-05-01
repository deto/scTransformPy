# scTransformPy
A Python implementation of of the scTransform method

Based on the R package [sctransform](https://github.com/ChristophH/sctransform) originally by Christoph Hafemeister

- [Hafemeister, C. & Satija, R. Normalization and variance stabilization of single-cell RNA-seq data using regularized negative binomial regression. bioRxiv 576827 (2019). doi:10.1101/576827](https://www.biorxiv.org/content/10.1101/576827v1)

Currently this supports basic functionality - variance stabilizing transform of UMI count data based on a general linear model and kernel-regularized parameters.

## Installation

```
pip install git+https://github.com/deto/scTransformPy.git
```

## Usage

```
import scTransform

residuals = scTransform.vst(
    umi, latent_var=['log_umi'], cell_attr
)
```

## Feature comparison with original R package

Supported:

- cell\_attr
- latent\_var

Unsupported:

- batch variables (batch\_var)
- non-regularized latent variables (latent\_var\_nonreg)
- gene sub-sampling (n\_genes)
- cell sub-sampling (n\_cells)
- alternate GLM fitting procedures (only the default, method="poisson" supported)
- sparse input umi matrix
