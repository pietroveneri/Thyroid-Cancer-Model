#%%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, mean_squared_error, roc_curve, auc, precision_recall_curve
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#%%
data = load_breast_cancer()
dt = pd.DataFrame(data.data, columns=data.feature_names)
X = data.data
y = data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)

#%%
models = {
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier()
}
#%%
cv_results = {}

for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    cv_results[name] = scores.mean()
    print(f"{name} Cross-Validation Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# %%
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs  = 1)
grid_search.fit(X_train, y_train)
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_:.3f}")

#%%

best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1] # probability of the positive class

#%%
plt.figure(figsize = (10,8))
cm = confusion_matrix(y_test, y_pred)
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt = 'd', cmap = 'Blues')
plt.title('Confusion Matrix (Counts)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

#%%

plt.figure(figsize = (8,6))
sns.heatmap(cm_percentage, annot = True, fmt='.1f', cmap= 'Blues')
plt.title('Confusion Matrix (Percentages)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout
plt.show()

# %%

# ROC curve

plt.figure(figsize=(8, 6 ))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange', lw = 2, label = f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color = 'navy', lw=2, linestyle ='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

#%%

# Precision Recall Curve

plt.figure(figsize = (10,6))
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

plt.plot(recall, precision, color = 'blue', lw = 2, label = f'PR curve (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()


#%%

# Error Analysis by Prediciton Probability

plt.figure(figsize = (10,6))
probabilities = pd.DataFrame({
    'True_Label': y_test,
    'Predicted_Probability': y_pred_proba
})

sns.histplot(data = probabilities, x ='Predicted_Probability', hue = 'True_Label',
             multiple = 'stack', bins = 20)
plt.title('Distribution of Prediction Probabilities')
plt.xlabel('Prediction Probability')
plt.ylabel('Count')
plt.show()

#%%

# Feature-wise Error Analysis

feature_importance = pd.DataFrame({
    'feature': data.feature_names,
    'importance' : best_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

top_features = feature_importance.head(5)['feature'].tolist()
plt.figure(figsize=(15, 10))
for i, feature in enumerate(top_features, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x='True_Label', y=feature, data=pd.DataFrame({
        'True_Label': y_test,
        feature: X_test[:, data.feature_names.tolist().index(feature)]
    }))
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()


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
