import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score  
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import RFECV
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns

u = 42

file_path = r'E:\Project\Finished Projects\RF+LR + ABC - Optimization\HCV_modified.xlsx'
df = pd.read_excel(file_path)


X = df.iloc[:, :12]  
y = df.iloc[:, -1]    

class_counts = y.value_counts()

# Plotting the class distribution
plt.figure(figsize=(10, 6))
sns.barplot(x=class_counts.index.astype(str), y=class_counts.values, palette='viridis')
plt.title('Class Distribution', fontsize=16)
plt.xlabel('Class', fontsize=16)
plt.ylabel('Number of Instances' , fontsize=16)
plt.xticks(rotation=45)  
plt.show()

# Normalize numerical features
numerical_columns = [col for col in X.columns if col != 'Sex']
scaler = MinMaxScaler()
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

# Label encoding
label_encoder = LabelEncoder()
X['Sex'] = label_encoder.fit_transform(X['Sex'])

# Map target classes to integers
target_mapping = {'Blood Donor': 0,
    'suspect Blood Donor': 4,
    'Cirrhosis': 3,
    'Fibrosis': 2,
    'Hepatitis': 1}
y = y.map(target_mapping)

# Reverse mapping 
reverse_target_mapping = {v: k for k, v in target_mapping.items()}

# show two decimals
pd.options.display.float_format = '{:.2f}'.format


rf = RandomForestClassifier(n_estimators=1000, random_state=u)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=u)

# Lists to accumulate results
y_true_list = []  
y_pred_list = [] 
fold_accuracies_rf = []  

# Perform cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X, y), start=1):

    X_train_fold, y_train_fold = X.iloc[train_index], y.iloc[train_index]
    X_test_fold, y_test_fold = X.iloc[test_index], y.iloc[test_index]
    
    rf.fit(X_train_fold, y_train_fold)
    y_pred_rf = rf.predict(X_test_fold)
    
    # Store true and predicted labels
    y_true_list.extend(y_test_fold)
    y_pred_list.extend(y_pred_rf)
    
    # Calculate accuracy for this fold
    fold_accuracy = accuracy_score(y_test_fold, y_pred_rf)
    fold_accuracies_rf.append(fold_accuracy)

# Calculate average accuracy across all folds
average_accuracy_rf = np.mean(fold_accuracies_rf)

print(f"Average Fold Accuracy for Random Forest: {average_accuracy_rf:.2f}")

# Plot accuracy across folds
fold_numbers = list(range(1, len(fold_accuracies_rf) + 1))
plt.figure(figsize=(10, 5))
plt.plot(fold_numbers, fold_accuracies_rf, marker='o', linestyle='-', color='b', label='Accuracy per Fold')
plt.axhline(average_accuracy_rf, color='r', linestyle='--', label=f'Average Accuracy ({average_accuracy_rf:.2f})')
plt.xticks(fold_numbers)
plt.xlabel('Fold Number',fontsize=14)
plt.ylabel('Accuracy',fontsize=14)
plt.title('Random Forest Accuracy Across Folds',fontsize=14)
plt.grid(True)
plt.legend()
plt.show()

# Generate confusion matrix 
total_confusion_matrix_rf = confusion_matrix(y_true_list, y_pred_list)
print("Total Confusion Matrix:")
print(total_confusion_matrix_rf)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(total_confusion_matrix_rf, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.title('Confusion Matrix for Random Forest Model')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# classification report
class_report = classification_report(y_true_list, y_pred_list)
print("Classification Report:")
print(class_report)

#**************************second part*****************

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=u)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Reshuffle the resampled data
X_resampled, y_resampled = shuffle(X_resampled, y_resampled, random_state=u)

# Perform RFE for feature selection
log_reg = LogisticRegression(max_iter=1000, random_state=u)
rfe = RFECV(estimator=log_reg, step=1, cv=StratifiedKFold(5), scoring='accuracy')
X_rfe = rfe.fit_transform(X_resampled, y_resampled)  #contains selected featues
X_rfe_df = pd.DataFrame(X_rfe, columns=X.columns[rfe.support_])

# Print optimal number of features
optimal_num_features = rfe.n_features_
print(f"Optimal number of features: {optimal_num_features}")

# Get selected features for predictions on the original data
selected_features = X.columns[rfe.support_]
X_original_selected = X[selected_features]


confusion_matrices = []
classification_reports = []
fold_accuracies_log = [] 

# Perform cross-validation, training on resampled data and predicting on original data
for fold_number, (train_index, _) in enumerate(skf.split(X_rfe_df, y_resampled), start=1):
    X_train_fold, y_train_fold = X_rfe_df.iloc[train_index], y_resampled.iloc[train_index]
    
    log_reg.fit(X_train_fold, y_train_fold)
    
    y_pred = log_reg.predict(X)
    
    # Generate confusion matrix 
    cm = confusion_matrix(y, y_pred)
    confusion_matrices.append(cm)
    
    # Calculate and store classification report as a dictionary for this fold
    report = classification_report(y, y_pred, output_dict=True)
    classification_reports.append(report)
    
    # Store accuracy for this fold
    fold_accuracy_log = report["accuracy"]  # Accuracy is stored as a float in the report
    fold_accuracies_log.append((fold_number, fold_accuracy_log))  # Store fold number and accuracy as a tuple

# Extract accuracies and calculate the average accuracy
accuracies_log = [accuracy for _, accuracy in fold_accuracies_log]  
average_accuracy_log = np.mean(accuracies_log)  

print(f"Average Fold Accuracy for Logistic Regression: {average_accuracy_log:.2f}")

# Plot accuracy across folds
fold_numbers_log, accuracies_log = zip(*fold_accuracies_log)  
plt.figure(figsize=(10, 5))
plt.plot(fold_numbers_log, accuracies_log, marker='o', linestyle='-', color='b', label='Accuracy per Fold')
plt.axhline(average_accuracy_log, color='r', linestyle='--', label=f'Average Accuracy ({average_accuracy_log:.2f})')
plt.xticks(fold_numbers_log)
plt.xlabel('Fold Number',fontsize=14)
plt.ylabel('Accuracy',fontsize=14)
plt.title('Logistic Regression Accuracy Across Folds (Average: {:.2f})'.format(average_accuracy_log),fontsize=14)
plt.grid(True)
plt.legend()
plt.show()

# Average the confusion matrices across folds
avg_confusion_matrix = np.mean(confusion_matrices, axis=0)

# Average classification report metrics across folds, excluding accuracy
metrics = ["precision", "recall", "f1-score", "support"]
avg_report = {}
for label in classification_reports[0].keys():  # Iterate over class labels and overall metrics
    if label == "accuracy":  # Handle accuracy separately as it’s a single float
        avg_report["accuracy"] = np.mean([fold_report["accuracy"] for fold_report in classification_reports])
    else:
        avg_report[label] = {}
        for metric in metrics:
            avg_report[label][metric] = np.mean([fold_report[label][metric] for fold_report in classification_reports])

# Convert average report to a DataFrame, round to two decimal places, and transpose
avg_report_df = pd.DataFrame(avg_report).round(2).T

#average classification report and confusion matrix
print("Average Classification Report:\n", avg_report_df)
print("\nAverage Confusion Matrix:\n", np.round(avg_confusion_matrix, 2))
plt.figure(figsize=(10, 7))
sns.heatmap(avg_confusion_matrix, annot=True, fmt=".2f", cmap="Blues", cbar=True,
            xticklabels=reverse_target_mapping.values(), yticklabels=reverse_target_mapping.values())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Average Confusion Matrix')
plt.show()

# ******Plot accuracy across folds for both models in one figure*********#
plt.figure(figsize=(10, 5))
# Plot accuracy for Random Forest
plt.plot(fold_numbers, fold_accuracies_rf, marker='o', linestyle='-', color='b', label='Random Forest Accuracy per Fold')
# Plot accuracy for Logistic Regression
plt.plot(fold_numbers_log, accuracies_log, marker='o', linestyle='-', color='g', label='Logistic Regression Accuracy per Fold')

# Plot average accuracy lines
plt.axhline(average_accuracy_rf, color='r', linestyle='--', label=f'RF Average Accuracy ({average_accuracy_rf:.2f})')
plt.axhline(average_accuracy_log, color='orange', linestyle='--', label=f'Logistic Regression Average Accuracy ({average_accuracy_log:.2f})')
plt.xticks(fold_numbers)
plt.xlabel('Fold Number',fontsize=14)
plt.ylabel('Accuracy',fontsize=14)
plt.title('Model Accuracy Across Folds',fontsize=14)
plt.grid(True)
plt.legend()
plt.show()

# Reset pandas float format for any future display requirements
pd.reset_option('display.float_format')

# Array to store probability predictions for each fold on original X
proba_predictions = np.zeros((X.shape[0], len(np.unique(y))))  # shape: (num_samples, num_classes)

# Perform cross-validation, training on resampled data and predicting probabilities on original data
for train_index, _ in skf.split(X_rfe_df, y_resampled):
    X_train_fold, y_train_fold = X_rfe_df.iloc[train_index], y_resampled.iloc[train_index]

    log_reg.fit(X_train_fold, y_train_fold)
    
    # Predict probabilities on the original data
    proba_predictions += log_reg.predict_proba(X)

# Average the probabilities across folds
proba_predictions /= skf.get_n_splits()


# Make 2D array  for rf
rf_proba_cv = cross_val_predict(rf, X, y, cv=skf, method='predict_proba')

def blend_predictions(rf_proba_cv, log_reg_proba, alpha):
    return alpha * rf_proba_cv + (1 - alpha) * log_reg_proba

#  ABC algorithm
def abc_algorithm(rf_proba_cv, log_reg_proba, y_true, n_bees=50, max_iter=7000, L=0.7, U=0.9):
    best_accuracy = 0
    best_alpha = L 

    # ABC loop
    for iteration in range(max_iter):
        for bee in range(n_bees):
  
            alpha = np.random.uniform(L, U)
            
            blended_predictions = np.argmax(blend_predictions(rf_proba_cv, log_reg_proba, alpha), axis=1)

            accuracy = np.mean(blended_predictions == y_true)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_alpha = alpha

    return best_alpha, best_accuracy

best_alpha, best_accuracy = abc_algorithm(rf_proba_cv, proba_predictions, y, n_bees=50, max_iter=7000, L=0.7, U=0.9)

print(f"\nBest Alpha: {best_alpha}, Best Accuracy: {best_accuracy}")

# Make final predictions using the optimal alpha from the ABC algorithm
final_blended_proba = blend_predictions(rf_proba_cv, proba_predictions, best_alpha)
final_predictions = np.argmax(final_blended_proba, axis=1)

print("\nEnsemble Model Classification Report (using ABC algorithm):")
print(classification_report(y, final_predictions, target_names=list(reverse_target_mapping.values())))

conf_matrix_ensemble = confusion_matrix(y, final_predictions)
print("Ensemble Model Confusion Matrix (using ABC algorithm):")
print(conf_matrix_ensemble)

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_ensemble, annot=True, fmt='d', cmap='Blues')
plt.title('Ensemble Model Confusion Matrix (using ABC algorithm)',fontsize=14)
plt.xlabel('Predicted',fontsize=14)
plt.ylabel('Actual',fontsize=14)
plt.show()

