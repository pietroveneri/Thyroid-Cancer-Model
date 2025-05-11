

# Thyroid Cancer Prediction Model

This project explores the use of machine learning to predict the recurrence of thyroid cancer based on clinical and pathological patient data.
- **Preprocessing**: Label encoding, standardization, PCA, and SMOTE/undersampling.
- **Model Comparison**: Evaluated 11 classifiers using 5-fold cross-validation.
- **Final Model**: AdaBoost selected via cross-validation and tuned using GridSearchCV.
- **Metrics & visualization**: Accuracy, confusion matrix, feature importances exc..

## Getting Started

```bash
# Clone the repo
git clone https://github.com/bitpest/Thyroid-Cancer-Model.git
cd Thyroid-Cancer-Model

# Install dependencies
pip install -r requirements.txt

# Run the model
python ThyroidCancerModel.py
