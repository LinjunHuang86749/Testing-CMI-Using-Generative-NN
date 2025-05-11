# Testing Conditional Mean Independence Using Generative Neural Networks

This repository contains **code** of the numerical experiments of ICML 2025 paper [Testing Conditional Mean Independence Using Generative Neural Networks](https://arxiv.org/abs/2501.17345),[Linjun Huang](https://linjun-site.netlify.app/), [Yi Zhang](https://orcid.org/0009-0000-4049-7347), [Yun Yang](https://sites.google.com/site/yunyangstat/) and [Xiaofeng Shao](https://scholar.google.com/citations?user=Z_ZD6A4AAAAJ&hl=en).

Conditional mean independence (CMI) testing is a fundamental tool for model simplification and assessing variable importance. However, existing test procedures suffer from severe performance deterioration in high dimensional setting.  We propose a new test procedure, basing on a novel CMI measure and neural networks, that has strong empirical performance in scenarios with high-dimensional covariates and response variable.  Our test can help in improving model efficiency, accuracy, and interpretability for many machine learning applications.


## Description

* Repository `Example_A1_A2` contains code used in the experiments for the **3. Simulation Results**.
* Repository `Facial_Expression_Application` contains code used in the experiments for the **4.1. Facial expression recognition**.
* Repository `Facial_Age_Application` contains code used in the experiments for the **4.2. Facial age estimation**.

## Dependencies

The following packages and versions were used in this project:

- Python:  3.11.12
- torch:  2.6.0+cu124
- numpy: 2.0.2
- scipy: 1.15.2
- xgboost: 2.1.4
- sklearn: 1.6.1
- tqdm: 4.67.1
