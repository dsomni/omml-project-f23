# OMML IU F23 Project: Loopless stochastic methods

by Dmitry Beresnev (<d.beresnev@innopolis.university>)
and Vsevolod Klyushev Dmitry Beresnev (<v.klyushev@innopolis.university>)

## Introduction

This repository contains project data for Optimization Methods for Machine Learning F23 IU course

## Requirements

Code was tested on Windows 11 and Python 3.11

All the requirement packages are listed in the file `requirements.txt`. In case you use the `pipenv` package, there is also `Pipfile` in the root of the project.

## Before start

Install all the packages from _requirements.txt_ using `pip install -r requirements.txt` or using **pipenv**: `pipenv install` or `pipenv install -d` for dev packages.

Optionally, you can run `bash setup_precommit.sh` to setup pre-commit hook for GitHub for code formatting using [ruff](https://docs.astral.sh/ruff/).

I also highly recommend to read reports in corresponding `reports` folder to fully understand context and purpose of some files and folders.

## Repository structure

```text
├── README.md       # The top-level README
├── Pipfile         # File with dependencies for pipenv
├── pyproject.toml  # Formatter and linter settings
├── setup_precommit.sh  # Script for creating pre-commit GitHub hook
|
│
├── notebooks   #  Jupyter notebooks
│   └── methods_comparison.ipynb  #  Compares methods from reference papers
│
├── references      # Reference papers
│
├── reports                 # Generated analysis as HTML, PDF, LaTeX, etc.
│   └── papers_overview.md  # Overview of reference papers
│
└── requirements.txt  # The requirements file for reproducing the analysis environment
                        generated with 'pip freeze › requirements.txt'
```

## References

### Papers

- [Don’t Jump Through Hoops and Remove Those Loops: SVRG and Katyusha are Better Without the Outer Loop](https://proceedings.mlr.press/v117/kovalev20a.html)
- [PAGE: A Simple and Optimal Probabilistic Gradient Estimator for Nonconvex Optimization](https://proceedings.mlr.press/v139/li21a.html)

### Books

- [Methods Of Convex Optimization by Yuri Nesterov, 2010](https://mipt.ru/dcam/upload/abb/nesterovfinal-arpgzk47dcy.pdf)

## Contacts

In case of any questions you can contact me via email <d.beresnev@innopolis.university> or Telegram **@d.flip.floppa**
