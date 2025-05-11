# README.md

![Project Status](https://img.shields.io/badge/status-active-brightgreen)
![Python Version](https://img.shields.io/badge/python-3.13%2B-blue)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

## Thyroid Cancer Recurrence Prediction

This project explores the use of machine learning to predict the recurrence of thyroid cancer based on clinical and pathological patient data.

## Table of Contents

- [Summary](#summary)
- [Model Highlights](#modelhighlights)
- [Key Results](#keyresults)
- [Installation](#installation)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Summary

An independent undergraduate research project focused on building a predictive model for thyroid cancer recurrence using structured patient data (n=383). The final model, based on AdaBoost, achieved 96% accuracy and AUC 0.98 on a hold-out test set. The pipeline includes preprocessing, model selection, hyperparameter tuning, ensemble comparisons, and rigorous internal validation.

---

## Model Highlights

- Preprocessing: Label encoding, standardization, PCA, and SMOTE/undersampling.
- Model Comparison: Evaluated 11 classifiers using 5-fold cross-validation.
- Final Model: AdaBoost selected via cross-validation and tuned using GridSearchCV.
- Validation:
   - Hold-out test evaluation
   - 5-fold CV on multiple metrics
   - Bootstrap resampling (1,000 iterations)
   - ROC curves and AUC
   - Calibration and learning/validation curves

---

## Key Results

- Test Accuracy: 96%
- AUC: 0.98
- Precision/Recall: Balanced across classes (~92–98%)
- Calibration: Good alignment between predicted probabilities and true outcomes
- Overfitting: No significant gap between training and test performance
  
---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/bitpest/Thyroid-Cancer-Model.git
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
  python ThyroidCancer.py

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
This model was developed purely for academic purposes and is not intended for clinical deployment. All analyses are based on internal validation using a single dataset.

---

## Contact

Created by Pietro Veneri. Reach out via Github or email: pietro.veneri72@gmail.com
