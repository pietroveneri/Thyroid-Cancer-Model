#%%
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve, validation_curve, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve, validation_curve, KFold
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import VotingClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


#%%

# Load and preprocess data
df = pd.read_csv('Thyroid_Diff.csv')
df.head()

# Identify categorical columns
categorical_columns = ['Gender', 'Smoking', 'Hx Smoking', 'Hx Radiothreapy', 
                      'Thyroid Function', 'Physical Examination', 'Adenopathy',
                      'Pathology', 'Focality', 'Risk', 'T', 'N', 'M', 'Stage',
                      'Response', 'Recurred']

# Store original categories for 'Recurred' before conversion
recurred_categories = df['Recurred'].unique()

# Convert categorical columns to category type
for col in categorical_columns:
    df[col] = df[col].astype('category')

# Convert categorical variables to numerical codes
for col in categorical_columns:
    df[col] = df[col].cat.codes

#%%
# Prepare features and target
X = df.drop('Recurred', axis=1)  # All columns except the target
y = df['Recurred']  # Target variable

# Split the data first
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Initialize and fit scaler only on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Transform test data using the same scaler

# Now we can proceed with the scaled data
X_train, X_test = X_train_scaled, X_test_scaled

plt.figure(figsize=(18,8))
sns.countplot(x=y)
plt.title("Disease Class Distribution Before Resampling")
plt.xticks(rotation=90)
plt.show()

#%%

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'SVM': SVC(random_state=42, probability=True),  # Added probability=True for predict_proba
    'Naive Bayes': GaussianNB(),
    'Neural Network': MLPClassifier(max_iter=1000, random_state=42),
    'LDA': LinearDiscriminantAnalysis(),
    'Extra Trees': ExtraTreesClassifier(random_state=42)
}

# Perform cross-validation for each model
cv_results = {}
print("\nCross-Validation Results (5-fold):")
print("-" * 50)

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_results[name] = {
        'mean_score': scores.mean(),
        'std_score': scores.std()
    }
    print(f"{name:20} Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# Sort models by mean accuracy
sorted_results = sorted(cv_results.items(), key=lambda x: x[1]['mean_score'], reverse=True)

print("\nModels ranked by accuracy:")
print("-" * 50)
for name, scores in sorted_results:
    print(f"{name:20} Accuracy: {scores['mean_score']:.3f} (+/- {scores['std_score'] * 2:.3f})")

# Get the best performing model
best_model_name = sorted_results[0][0]
best_model = models[best_model_name]

print(f"\nBest performing model: {best_model_name}")

# Fine-tune the best model
if best_model_name == 'AdaBoost':
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.5, 1]
    }
elif best_model_name == 'SVM':
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'class_weight': [None, 'balanced']
    }
elif best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
else:
    param_grid = {}  # Use default parameters for other models

if param_grid:
    print(f"\nFine-tuning {best_model_name}...")
    grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best CV Score: {grid_search.best_score_:.3f}")
    best_model = grid_search.best_estimator_

# Train and evaluate the best model
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
 
# Print comprehensive evaluation metrics
print("\nModel Evaluation Metrics:")
print("-" * 50)
print(classification_report(y_test, y_pred, target_names=recurred_categories))
#
# Plot confusion matrix
plt.figure(figsize=(10, 8)) 
cm = confusion_matrix(y_test, y_pred)
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=recurred_categories,
            yticklabels=recurred_categories)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
#%%
# Plot percentage confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=recurred_categories,
            yticklabels=recurred_categories)
plt.title(f'Confusion Matrix (Percentages) - {best_model_name}')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
#%%
# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve - {best_model_name}')
plt.legend(loc="lower right")
plt.show()

# %%

# Feature Selection and Engineering
print("\nModel Improvement Analysis:")
print("-" * 50)

# 1.1. Feature Selection
print("\n1.1. Feature Selection Analysis:")
selector = SelectKBest(f_classif, k='all')
selector.fit(X_train, y_train)
feature_scores = pd.DataFrame({
    'Feature': X.columns,  # Use all features
    'Score': selector.scores_
})
feature_scores = feature_scores.sort_values('Score', ascending=False)
print("\nFeature Importance Scores:")
print(feature_scores)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Score', y='Feature', data=feature_scores)
plt.title('Feature Importance Scores')
plt.tight_layout()
plt.show()

# %% 

# 1.2. PCA Analysis
print("\n1.2. PCA Analysis:")
pca = PCA()
X_pca = pca.fit_transform(X_train)
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, 'bo-')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Cumulative Explained Variance')
plt.show()

# %% 

# 1.3. Class Distribution Analysis
print("\n1.3. Class Distribution Analysis:")
class_dist = pd.Series(y_train).value_counts()
print("\nClass Distribution:")
print(class_dist)

plt.figure(figsize=(8, 6))
sns.barplot(x=class_dist.index, y=class_dist.values)
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# %% 
# 1.4. Try different sampling strategies
print("\n1.4. Testing Different Sampling Strategies:")

# Create sampling strategies
samplers = {
    'Original': None,
    'SMOTE': SMOTE(random_state=42),
    'RandomUnderSampler': RandomUnderSampler(random_state=42)
}

# Test each sampling strategy
for name, sampler in samplers.items():
    if sampler is None:
        X_resampled, y_resampled = X_train, y_train
    else:
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    
    # Train and evaluate
    model = best_model.__class__(**best_model.get_params())
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(f"{name:20} Accuracy: {score:.3f}")
# %% 
# 1.5. Try ensemble methods
print("\n1.5. Testing Ensemble Methods:")

# Create ensemble of best models
top_models = {name: model for name, model in models.items() 
             if name in [m[0] for m in sorted_results[:3]]}

ensemble = VotingClassifier(
    estimators=[(name, model) for name, model in top_models.items()],
    voting='soft'
)

ensemble.fit(X_train, y_train)
y_pred_ensemble = ensemble.predict(X_test)
ensemble_score = accuracy_score(y_test, y_pred_ensemble)
print(f"Ensemble Accuracy: {ensemble_score:.3f}")

# Compare ensemble with best single model
print(f"Best Single Model ({best_model_name}) Accuracy: {accuracy_score(y_test, y_pred):.3f}")
# %%
# 1.6. Cross-validation with different metrics
print("\n1.6. Cross-validation with different metrics:")
scoring_metrics = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

for metric_name, metric in scoring_metrics.items():
    scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring=metric)
    print(f"{metric_name:10} Score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
# %%
# Recommendations based on analysis
print("\nRecommendations for Model Improvement:")
print("-" * 50)

# Feature selection recommendation
print("\n1.7. Feature Selection:")
print(f"Consider using only the top {sum(cumulative_variance < 0.95)} features that explain 95% of variance")
print("Top 5 most important features:")
print(feature_scores.head().to_string())


# %% 
# 2.0. Overfitting Analysis and Validation
print("\nOverfitting Analysis and Validation:")
print("-" * 50)

# 2.1. Learning Curves
print("\n2. 1. Learning Curves Analysis:")
train_sizes, train_scores, test_scores = learning_curve(
    best_model, X_train, y_train,
    cv=5,
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training score')
plt.plot(train_sizes, test_mean, label='Cross-validation score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
plt.xlabel('Training Examples')
plt.ylabel('Score')
plt.title('Learning Curves')
plt.legend(loc='best')
plt.grid(True)
plt.show()
# %%
# 2.2. Validation Curves for key parameters
print("\n2.2. Validation Curves Analysis:")

if best_model_name == 'AdaBoost':
    param_name = 'n_estimators'
    param_range = [50, 100, 200, 300, 400, 500]
elif best_model_name == 'Random Forest':
    param_name = 'n_estimators'
    param_range = [50, 100, 200, 300, 400, 500]
elif best_model_name == 'SVM':
    param_name = 'C'
    param_range = [0.1, 1, 10, 100, 1000]
elif best_model_name == 'Gradient Boosting':
    param_name = 'learning_rate'
    param_range = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
else:
    param_name = 'max_iter'
    param_range = [100, 500, 1000, 2000, 3000]

# Ensure y_train is a numpy array
y_train_array = y_train.values if hasattr(y_train, 'values') else y_train

train_scores, test_scores = validation_curve(
    best_model, X_train, y_train_array,
    param_name=param_name,
    param_range=param_range,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(param_range, train_mean, label='Training score')
plt.plot(param_range, test_mean, label='Cross-validation score')
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.1)
plt.xlabel(param_name)
plt.ylabel('Score')
plt.title(f'Validation Curve for {param_name}')
plt.legend(loc='best')
plt.grid(True)
plt.show()
# %%
# After model training and before final evaluation
print("\nDetailed Prediction Analysis and Visualizations:")
print("-" * 50)
# %%
# 2.3. Prediction Probability Distribution
print("\n2.3. Prediction Probability Distribution Analysis:")
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

plt.figure(figsize=(12, 6))
plt.hist(y_pred_proba, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
plt.title('Distribution of Prediction Probabilities')
plt.xlabel('Probability of Recurrence')
plt.ylabel('Count')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#%%

# 2.4. Confidence vs Accuracy Analysis
print("\n2.4. Confidence vs Accuracy Analysis:")
confidence_bins = np.linspace(0, 1, 11)
accuracy_by_confidence = []

for i in range(len(confidence_bins)-1):
    mask = (y_pred_proba >= confidence_bins[i]) & (y_pred_proba < confidence_bins[i+1])
    if np.sum(mask) > 0:
        accuracy = np.mean(y_test[mask] == y_pred[mask])
        accuracy_by_confidence.append(accuracy)
    else:
        accuracy_by_confidence.append(np.nan)

plt.figure(figsize=(10, 6))
plt.plot(confidence_bins[:-1], accuracy_by_confidence, 'bo-', linewidth=2)
plt.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect Calibration')
plt.title('Model Confidence vs Accuracy')
plt.xlabel('Prediction Confidence')
plt.ylabel('Actual Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#%%
# 2.5. Prediction Calibration Plot
print("\n2.5. Prediction Calibration Analysis:")
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
plt.figure(figsize=(10, 6))
plt.plot(prob_pred, prob_true, 'bo-', label='Model')
plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
plt.title('Calibration Plot')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('True Probability')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#%%
# 3. K-Fold Cross Validation with multiple metrics
print("\n3. K-Fold Cross Validation with Multiple Metrics:")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
metrics = {
    'accuracy': accuracy_score,
    'precision': precision_score,
    'recall': recall_score,
    'f1': f1_score
}

cv_results = {metric: [] for metric in metrics.keys()}

# Ensure y_train is a numpy array
y_train_array = y_train.values if hasattr(y_train, 'values') else y_train

for train_index, test_index in kf.split(X_train):
    # Use direct numpy array indexing
    X_train_fold = X_train[train_index]
    X_test_fold = X_train[test_index]
    y_train_fold = y_train_array[train_index]
    y_test_fold = y_train_array[test_index]
    
    # Train and predict
    best_model.fit(X_train_fold, y_train_fold)
    y_pred_fold = best_model.predict(X_test_fold)
    
    # Calculate metrics
    for metric_name, metric_func in metrics.items():
        score = metric_func(y_test_fold, y_pred_fold)
        cv_results[metric_name].append(score)

# Print results
print("\nCross-validation results (mean ± std):")
for metric_name, scores in cv_results.items():
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"{metric_name:10}: {mean_score:.3f} ± {std_score:.3f}")
# %%
# 4. Bootstrap Validation
print("\n4. Bootstrap Validation:")
n_iterations = 1000
bootstrap_scores = []

# Ensure y_train and y_test are numpy arrays
y_train_array = y_train.values if hasattr(y_train, 'values') else y_train
y_test_array = y_test.values if hasattr(y_test, 'values') else y_test

for i in range(n_iterations):
    # Create bootstrap sample
    indices = np.random.randint(0, len(X_train), len(X_train))
    X_bootstrap = X_train[indices]
    y_bootstrap = y_train_array[indices]
    
    # Train and evaluate
    best_model.fit(X_bootstrap, y_bootstrap)
    score = best_model.score(X_test, y_test_array)
    bootstrap_scores.append(score)

# Calculate confidence intervals
confidence_interval = np.percentile(bootstrap_scores, [2.5, 97.5])
print(f"\nBootstrap 95% Confidence Interval: [{confidence_interval[0]:.3f}, {confidence_interval[1]:.3f}]")
# %%
# 5. Overfitting Analysis
print("\n5. Overfitting Analysis:")
train_score = best_model.score(X_train, y_train_array)
test_score = best_model.score(X_test, y_test_array)
print(f"Training Score: {train_score:.3f}")
print(f"Test Score: {test_score:.3f}")
print(f"Difference (Train - Test): {train_score - test_score:.3f}")

if train_score - test_score > 0.1:
    print("\nWARNING: Model might be overfitting!")
    print("Consider:")
    print("1. Reducing model complexity")
    print("2. Adding regularization")
    print("3. Collecting more training data")
    print("4. Using feature selection to reduce dimensionality")
else:
    print("\nModel shows good generalization!")
# %%
# 6. Final Model Evaluation with Confidence
print("\n6. Final Model Evaluation with Confidence:")
print(f"Model: {best_model_name}")
print(f"Average CV Score: {np.mean(cv_results['accuracy']):.3f} ± {np.std(cv_results['accuracy']):.3f}")
print(f"95% Confidence Interval: [{confidence_interval[0]:.3f}, {confidence_interval[1]:.3f}]")
print(f"Overfitting Indicator: {train_score - test_score:.3f}")

# Save the final model if it shows good generalization
if train_score - test_score <= 0.1:
    print("\nModel shows good generalization.")
    #import joblib
    #joblib.dump(best_model, 'thyroid_cancer_model.joblib')
else:
    print("\nModel needs improvement before deployment.")

# %%

'''

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, 
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS 
IN THE SOFTWARE

'''