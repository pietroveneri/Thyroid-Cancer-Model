# README.md

![Project Status](https://img.shields.io/badge/status-active-brightgreen)
![Python Version](https://img.shields.io/badge/python-3.13%2B-blue)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

## Project Title

**CancerCellModel**: An end-to-end breast cancer cell classification pipeline using scikit-learn and Python.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

This project demonstrates the full machine learning workflow on the UCI breast cancer dataset:

- Data loading & preprocessing with pandas and scikit-learn’s `StandardScaler`
- Model benchmarking: GaussianNB, SVM, Random Forest with cross-validation
- Hyperparameter tuning using `GridSearchCV`
- Model evaluation: accuracy, confusion matrix, feature importance visualization
- Error measurements: MSE, ROC curve, Feature-wise Error Analysis..

---

## Features

- Clean, modular Python script (`CancerCellModel.py`)
- Automated benchmarking of multiple classifiers
- Plots saved to `docs/assets/images/`
- Configurable hyperparameters via command-line arguments

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows

3. Install dependencies
   ```bash
   pip install -r requirements.txt

---

## Usage

- Run the classification script with default parameters:
  ```bash
  python CancerCellModel.py

---

## Requirements

- Python 3.13 or higher
- pandas
- scikit-learn
- matplotlib
- seaborn
- numpy
  
---

## Contributing

Contributions are welcome! Please fork the repo, create a branch for your feature or bug fix, and submit a pull request.

---

## License

This project is licensed under the MIT License.

---

## Contact

Created by Pietro Veneri. Reach out via Github or email: pietro.veneri72@gmail.com
