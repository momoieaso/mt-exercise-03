import os
import pandas as pd
import matplotlib.pyplot as plt
import re

# Set paths based on your directory structure
log_dir = '../models/logs'
output_dir = '../models/ppls'

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

dropout_values = [0, 0.1, 0.3, 0.6, 0.8]
epochs = 40

def parse_log_file(logfile_path):
    """ Parses the log file to extract perplexity information. """
    train_ppl = []
    valid_ppl = []
    test_ppl = None
    with open(logfile_path, 'r') as file:
        for line in file:
            # Adjust regex to match the specific log entries
            train_match = re.search(r'\| epoch\s+\d+\s+\|\s+ppl\s+(\d+\.\d+)', line)
            if train_match:
                train_ppl.append(float(train_match.group(1)))

            valid_match = re.search(r'\| end of epoch\s+\d+\s+\|\s+valid ppl\s+(\d+\.\d+)', line)
            if valid_match:
                valid_ppl.append(float(valid_match.group(1)))

            test_match = re.search(r'\| End of training\s+\|\s+test ppl\s+(\d+\.\d+)', line)
            if test_match:
                test_ppl = float(test_match.group(1))

    return train_ppl, valid_ppl, test_ppl


def create_table(dropout_values, epoch_count, parse_function, index_type='train'):
    columns = [f'Dropout {dropout}' for dropout in dropout_values]
    data = {col: [] for col in columns}
    all_empty = True

    for dropout in dropout_values:
        filepath = f'{log_dir}/perplexity_dropout_{dropout}.log'
        results = parse_function(filepath)
        if index_type == 'test':
            # For test data, only a single value is needed, not a list per dropout value
            data[f'Dropout {dropout}'] = [results[2]] if results[2] is not None else [float('nan')]
            all_empty = False
        else:
            if results[0]:  # Train or valid data
                data_values = results[0] if index_type == 'train' else results[1]
                data[f'Dropout {dropout}'] = data_values[:epoch_count]
                all_empty = False
            else:
                print(f"No data found for {dropout}, skipping.")

    if all_empty:
        print("No data available to create DataFrame.")
        return pd.DataFrame()  # Return an empty DataFrame if no data was found

    # Adjust index for test which only needs a single entry
    index = ['Test PPL'] if index_type == 'test' else [f'Epoch {i + 1}' for i in range(epoch_count)]
    return pd.DataFrame(data, index=index)



# Generate tables
train_perplexity = create_table(dropout_values, epochs, parse_log_file, 'train')
valid_perplexity = create_table(dropout_values, epochs, parse_log_file, 'valid')
test_perplexity = create_table(dropout_values, epochs, parse_log_file, 'test')

# Print and save tables to CSV
print("Train Perplexity:")
print(train_perplexity)
train_perplexity.to_csv(os.path.join(output_dir, 'Train_Perplexity.csv'))

print("Valid Perplexity:")
print(valid_perplexity)
valid_perplexity.to_csv(os.path.join(output_dir, 'Valid_Perplexity.csv'))

print("Test Perplexity:")
print(test_perplexity)
test_perplexity.to_csv(os.path.join(output_dir, 'Test_Perplexity.csv'))

def plot_perplexity(df, title, filename=None):
    """ Plots perplexity data from a DataFrame and optionally saves it as a file. """
    plt.figure(figsize=(10, 5))
    for column in df.columns:
        plt.plot(range(1, len(df.index) + 1), df[column], label=column)
    plt.title(title)
    plt.xlabel('Epoch' if 'Epoch' in df.index[0].__str__() else 'Index')
    plt.ylabel('Perplexity')
    plt.legend(title='Dropout Rate')
    plt.grid(True)

    plt.xticks(range(1, len(df.index) + 1), rotation=45)  
    plt.xlim(0, len(df.index) + 1)  
    
    plt.tight_layout()
    if filename:
        plt.savefig(os.path.join(output_dir, filename))  # Save the plot as an image file
    plt.close()  # Close the plot to free up resources

# Plotting
plot_perplexity(train_perplexity, 'Training Perplexity Across Epochs', 'train_perplexity.png')
plot_perplexity(valid_perplexity, 'Validation Perplexity Across Epochs', 'valid_perplexity.png')
