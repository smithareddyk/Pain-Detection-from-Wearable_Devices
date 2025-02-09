import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Function to load data from the CSV file
def load_data(file_path):
    # Read the first three columns with pandas
    column_names = ['System ID', 'Data Type', 'Class']
    data = pd.read_csv(file_path, header=None, usecols=[0, 1, 2], names=column_names)  
    # Creating a copy of the DataFrame
    data_primary = data.copy()
    # Read the file line by line for the 'Data' column to ensure matching row counts
    with open(file_path, 'r') as file:
        data_strings = [','.join(line.strip().split(',')[3:]) for line in file]        
        # If there is a header row, remove the first element
        if len(data_strings) > data_primary.shape[0]:
            data_strings.pop(0)
    # Add the concatenated data strings as a new 'Data' column to the DataFrame
    data_primary['Data'] = data_strings
    return data_primary

# Function to filter data based on the specified data type
def filter_data(data, data_type):
    # Check if a specific data type is requested and filter accordingly
    if data_type != 'all':
        if data_type == 'dia':
            return data[data['Data Type'] == 'BP Dia_mmHg']
        if data_type == 'sys':
            return data[data['Data Type'] == 'LA Systolic BP_mmHg']
        if data_type == 'eda':
            return data[data['Data Type'] == 'EDA_microsiemens']
        if data_type == 'res':
            return data[data['Data Type'] == 'Respiration Rate_BPM']
    return data

#Function to extract features from the data
def extract_features(data):
    data_copy = data.copy()
    # Applying the transformation
    data_copy['Data'] = data_copy['Data'].apply(lambda x: np.array(x.split(',')).astype(float))
    features = data_copy['Data'].apply(lambda x: pd.Series([x.mean(), x.var(), x.min(), x.max()]))
    features.columns = ['mean', 'variance', 'min', 'max']
    return features


#Function to train the model and evaluate its performance
def train_model(features, labels):
    # Initialize the RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    # Set up K-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    #kf = KFold(n_splits=10)
    # Variables to accumulate metrics across folds
    cm_total = np.zeros((2, 2), dtype=int)
    acc_scores, prec_scores, recall_scores = [], [], []

    # Cross-validation loop
    for train_index, test_index in kf.split(features):
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Accumulate performance metrics
        cm_total += confusion_matrix(y_test, predictions, labels=model.classes_)
        acc_scores.append(accuracy_score(y_test, predictions))
        prec_scores.append(precision_score(y_test, predictions, pos_label='Pain'))
        recall_scores.append(recall_score(y_test, predictions, pos_label='Pain'))

    # Calculate average performance metrics across all folds
    cm_avg = cm_total / 10.0
    acc_avg = np.mean(acc_scores)
    prec_avg = np.mean(prec_scores)
    recall_avg = np.mean(recall_scores)

    return cm_avg, acc_avg, prec_avg, recall_avg

# Main function to orchestrate the data loading, processing, and model training
def main(file_path, data_type):
    data = load_data(file_path)
    filtered_data = filter_data(data, data_type)
    features = extract_features(filtered_data)
    labels = filtered_data['Class']
    cm, acc, prec, recall = train_model(features, labels)
    #boxplot
    plt.figure(figsize=(8, 5))
    features.boxplot()
    plt.show()
    #linegraph for an instance
    data_instance = data[data['System ID'] == 'F001']
    # Separate data based on Data Type
    data_types = data_instance['Data Type'].unique()
    classes = data_instance['Class'].unique()

    # Define colors for each data type
    # Define colors for each data type and class combination
    colors = {
        ('BP Dia_mmHg', 'No Pain'): 'blue',
        ('EDA_microsiemens', 'No Pain'): 'green',
        ('LA Systolic BP_mmHg', 'No Pain'): 'red',
        ('Respiration Rate_BPM', 'No Pain'): 'purple',
        ('BP Dia_mmHg', 'Pain'): 'orange',
        ('EDA_microsiemens', 'Pain'): 'brown',
        ('LA Systolic BP_mmHg', 'Pain'): 'cyan',
        ('Respiration Rate_BPM', 'Pain'): 'magenta'
        }

    # Plot each data type with a unique color based on class
    plt.figure(figsize=(10, 6))
    for i, data_type in enumerate(data_types):
        df_type = data_instance[data_instance['Data Type'] == data_type]
        for index, row in df_type.iterrows():
            data_values = [float(val) for val in row['Data'].split(',')]
            color = colors[(data_type, row['Class'])]
            plt.plot(data_values, label=data_type + ' - ' + row['Class'], color=color)

    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.title('Physiological Data for F001')
    plt.legend()
    plt.grid(True)
    plt.show()
   
 
    # Print the performance metrics
    # The Accuracy, Precision, and Recall percentage is from 0-1, 0-means 0% and 1-means 100%
    print("Confusion Matrix:\n", cm)
    print("Accuracy: {:.2f}".format(acc))
    print("Precision: {:.2f}".format(prec))
    print("Recall: {:.2f}".format(recall))

# The below line is the entrypoint to the code
if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # This will help to read the command line arguments
    parser.add_argument('data_type', choices=['dia', 'sys', 'eda', 'res', 'all'])
    parser.add_argument('data_file', type=str)
    args = parser.parse_args()
    main(args.data_file, args.data_type)

