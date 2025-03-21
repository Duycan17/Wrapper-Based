import os
import pandas as pd
import scipy.io
import numpy as np

# Define the main folders and wrapper methods
main_folders = ['AB', 'DT', 'ET', 'KNN', 'LR', 'RF']
wrapper_methods = ['gndo', 'sma', 'hho', 'pfa', 'mrfo']
metrics = ['acc', 'auc', 'precision', 'recall', 'f1']

def read_matlab_file(file_path):
    """Read .mat file and extract metrics"""
    try:
        mat_data = scipy.io.loadmat(file_path)
        results = {}
        for metric in metrics:
            if metric in mat_data:
                results[metric] = mat_data[metric].item()
            else:
                results[metric] = np.nan
        return results
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return {metric: np.nan for metric in metrics}

# Create a list to store all results
all_results = []

# Base path for the input directory
base_path = '/kaggle/input/wrapper-result-duycun'

# Iterate through the folder structure
for classifier in main_folders:
    for wrapper in wrapper_methods:
        path = os.path.join(base_path, classifier, wrapper)
        
        if not os.path.exists(path):
            continue
            
        mat_files = [f for f in os.listdir(path) if f.endswith('.mat')]
        
        for mat_file in mat_files:
            file_path = os.path.join(path, mat_file)
            dataset_name = os.path.splitext(mat_file)[0]
            results = read_matlab_file(file_path)
            
            result_row = {
                'Classifier': classifier,
                'Wrapper': wrapper,
                'Dataset': dataset_name,
                **results
            }
            all_results.append(result_row)

# Create DataFrame from results
df = pd.DataFrame(all_results)

# Calculate average metrics for each classifier-wrapper combination
avg_by_classifier_wrapper = df.groupby(['Classifier', 'Wrapper'])[metrics].mean()
avg_by_classifier = df.groupby(['Classifier'])[metrics].mean()
avg_by_wrapper = df.groupby(['Wrapper'])[metrics].mean()

# Round all numeric values to 4 decimal places
df[metrics] = df[metrics].round(4)
avg_by_classifier_wrapper = avg_by_classifier_wrapper.round(4)
avg_by_classifier = avg_by_classifier.round(4)
avg_by_wrapper = avg_by_wrapper.round(4)

# Create Excel writer object with better formatting
output_path = 'results_summary.xlsx'
with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
    workbook = writer.book
    
    # Create formats
    header_format = workbook.add_format({
        'bold': True,
        'bg_color': '#D9E1F2',
        'border': 1
    })
    
    cell_format = workbook.add_format({
        'border': 1
    })
    
    # 1. Overview Sheet
    overview_df = pd.DataFrame({
        'Total Datasets': [len(df['Dataset'].unique())],
        'Total Classifiers': [len(main_folders)],
        'Total Wrapper Methods': [len(wrapper_methods)],
        'Total Experiments': [len(df)]
    }).T
    overview_df.to_excel(writer, sheet_name='Overview')
    
    # 2. Detailed Results Sheet - organized by classifier and wrapper
    df_sorted = df.sort_values(['Classifier', 'Wrapper', 'Dataset'])
    df_sorted.to_excel(writer, sheet_name='Detailed Results', index=False)
    
    # 3. Average Results by Classifier
    format_excel_sheet(writer, 'Avg by Classifier', avg_by_classifier)
    
    # 4. Average Results by Wrapper
    format_excel_sheet(writer, 'Avg by Wrapper', avg_by_wrapper)
    
    # 5. Average Results by Classifier-Wrapper
    format_excel_sheet(writer, 'Avg by Classifier-Wrapper', avg_by_classifier_wrapper)
    
    # 6. Individual Metric Sheets
    for metric in metrics:
        # Create pivot table for each metric
        pivot_df = df.pivot_table(
            values=metric,
            index='Dataset',
            columns=['Classifier', 'Wrapper'],
            aggfunc='first'
        ).round(4)
        
        # Add summary statistics at the bottom
        pivot_df.loc['Mean'] = pivot_df.mean()
        pivot_df.loc['Std'] = pivot_df.std()
        pivot_df.loc['Max'] = pivot_df.max()
        pivot_df.loc['Min'] = pivot_df.min()
        
        sheet_name = f'{metric.upper()} Results'
        format_excel_sheet(writer, sheet_name, pivot_df)

print(f"Results have been saved to '{output_path}'")
print(f"Total results processed: {len(all_results)}")
