import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os
import glob

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Define paths
results_path = '/Users/neftekhari/Documents/NeurIPS_2024_Workshop_on_Tackling_Climate_Change_with_Machine_Learning/results/Topthree'
true_file = '/Users/neftekhari/Documents/NeurIPS_2024_Workshop_on_Tackling_Climate_Change_with_Machine_Learning/label/image_labels.csv'
acc_dir = os.path.join(results_path, 'ACC')
cm_dir = os.path.join(results_path, 'CM')
comparison_dir = '/Users/neftekhari/Documents/NeurIPS_2024_Workshop_on_Tackling_Climate_Change_with_Machine_Learning/results/comparision'

# Create directories if they don't exist
os.makedirs(acc_dir, exist_ok=True)
os.makedirs(cm_dir, exist_ok=True)
os.makedirs(comparison_dir, exist_ok=True)

# Load the ground truth labels
df_true = pd.read_csv(true_file)

# Function to process each prediction file
def process_file(pred_file):
    # Load the prediction data
    df_pred = pd.read_csv(pred_file)

    # Merge the dataframes on 'Filename'
    df_merged = pd.merge(df_true, df_pred, on='Filename', how='inner')

    # Extract true labels and predicted classes
    true_labels = df_merged['Label']
    predicted_classes = df_merged['Top1 Prediction']
    top2_predictions = df_merged['Top2 Prediction']
    top3_predictions = df_merged['Top3 Prediction']

    # Get unique classes from true labels
    unique_classes = sorted(df_true['Label'].unique())

    # Create confusion matrix
    conf_matrix = confusion_matrix(list(true_labels), list(predicted_classes))

    # Calculate percentages for each class
    conf_matrix_percentage = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

    # Generate classification report
    classification_rep = classification_report(true_labels, predicted_classes, output_dict=True)

    # Convert classification report to DataFrame
    df_classification_report = pd.DataFrame(classification_rep).transpose()
    df_classification_report.reset_index(inplace=True)
    df_classification_report.rename(columns={'index': 'Label'}, inplace=True)

    # Save the classification report
    classification_report_file = os.path.join(acc_dir, os.path.basename(pred_file).replace('.csv', '_report.csv'))
    df_classification_report.to_csv(classification_report_file, index=False)

    # Calculate additional metrics
    accuracy = accuracy_score(true_labels, predicted_classes)
    precision = precision_score(true_labels, predicted_classes, average='weighted')
    recall = recall_score(true_labels, predicted_classes, average='weighted')
    f1 = f1_score(true_labels, predicted_classes, average='weighted')

    # Calculate Top-3 Accuracy
    top3_correct = df_merged.apply(lambda row: row['Label'] in [row['Top1 Prediction'], row['Top2 Prediction'], row['Top3 Prediction']], axis=1)
    top3_accuracy = top3_correct.mean()

    # Print metrics
    print(f'\nAccuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'Top-3 Accuracy: {top3_accuracy}')

    # Save additional metrics to a file
    metrics_file = os.path.join(acc_dir, os.path.basename(pred_file).replace('.csv', '_metrics.txt'))
    with open(metrics_file, 'w') as f:
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'Precision: {precision}\n')
        f.write(f'Recall: {recall}\n')
        f.write(f'F1 Score: {f1}\n')
        f.write(f'Top-3 Accuracy: {top3_accuracy}\n')

    # Plot confusion matrix using seaborn
    plt.figure(figsize=(14, 12))  # Adjust size to fit long class names
    sns.heatmap(conf_matrix_percentage, annot=True, fmt='.2f', cmap='Blues', xticklabels=unique_classes, yticklabels=unique_classes)
    plt.title('Confusion Matrix (Percentages)')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.yticks(rotation=0)  # Ensure y-axis labels are horizontal
    plt.tight_layout()  # Ensure the plot fits within the figure area
    conf_matrix_file = os.path.join(cm_dir, os.path.basename(pred_file).replace('.csv', '.png'))
    plt.savefig(conf_matrix_file, transparent=True)
    # plt.show()

    return df_classification_report

# Process all prediction files in the results path and compare the minority class performance
f1_scores = {}
minority_class = 'Zoo_other'  # Replace with your actual minority class label

for pred_file in glob.glob(os.path.join(results_path, '*_topthree.csv')):
    df_report = process_file(pred_file)
    f1_scores[pred_file] = df_report.loc[df_report['Label'] == minority_class, 'f1-score'].values[0]

# Save the comparison results
comparison_file = os.path.join(comparison_dir, 'comparison_results.txt')
with open(comparison_file, 'w') as f:
    for model, f1_score in f1_scores.items():
        f.write(f'{model}: {f1_score}\n')

best_model = max(f1_scores, key=f1_scores.get)
print(f'The best model for classifying the minority class ({minority_class}) is: {best_model} with an F1 score of {f1_scores[best_model]}')
with open(comparison_file, 'a') as f:
    f.write(f'\nThe best model for classifying the minority class ({minority_class}) is: {best_model} with an F1 score of {f1_scores[best_model]}\n')
