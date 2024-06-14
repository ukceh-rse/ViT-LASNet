import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def read_csv_files(folder_path):
    data_frames = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path, nrows=3, usecols=['Class Name', 'precision', 'recall', 'f1-score'])
            df['Filename'] = file_name
            data_frames.append(df)

    df_all = pd.concat(data_frames, ignore_index=True)
    return df_all


def create_radar_chart(df, class_name):
    models = df['Class Name'].unique()
    labels = df['Filename'].unique()

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for i, model in enumerate(models):
        values = df[(df['Class Name'] == model)][[class_name]].values.tolist()

        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=model)
        ax.fill(angles, values, alpha=0.1)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    # ax.set_xticklabels(labels)
    ax.set_title(f'Radar Chart for {class_name.capitalize()}')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    # plt.show()


folder_path = r'/Users/neftekhari/Documents/EDA-elastic-eggs-results/classification_reports'
np.random.seed(42)
# Read CSV files and create the combined DataFrame
df = read_csv_files(folder_path)

metrics = ['precision', 'recall', 'f1-score']
for metric in metrics:
    create_radar_chart(df, metric)
