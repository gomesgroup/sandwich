import os
import pandas as pd

def analyze_csv(file_path):
    data = pd.read_csv(file_path)

    def classify_row(row):
        if row['dot_pro_all_C_and_H'] <= 0 and row['Magnitude_16'] >= 2:
            return 'yes mode 6'
        else:
            return 'no'

    data['Look Here'] = data.apply(classify_row, axis=1)

    pos_sequences = []
    start_step = None
    count = 0

    for step, value in zip(data['Step'], data['Look Here']):
        if value == 'yes mode 6':
            if start_step is None:
                start_step = step
            count += 1
        elif count > 0:
            pos_sequences.append((count, start_step, step))
            count = 0
            start_step = None

    if count > 0:
        pos_sequences.append((count, start_step, data['Step'].iloc[-1]))

    # Sort sequences by the length, in descending order
    pos_sequences.sort(reverse=True, key=lambda x: x[0])

    # Create a new DataFrame for the positive sequences
    pos_sequence_data = [f"max yes={seq[0]}, start={seq[1]}, end={seq[2]}" for seq in pos_sequences]
    pos_sequence_df = pd.DataFrame({'Yes Count': pos_sequence_data})

    # Merge the original data with the 'Yes Count' DataFrame
    data = pd.concat([data, pos_sequence_df.reindex(data.index, fill_value='')], axis=1)

    return data

def main():
    input_dir = '/path'
    output_dir = '/path'
    os.makedirs(output_dir, exist_ok=True)

    # Process each CSV file
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.csv'):
            file_path = os.path.join(input_dir, file_name)
            print(f"Processing {file_path}...")
            modified_data = analyze_csv(file_path)
            output_path = os.path.join(output_dir, file_name)
            modified_data.to_csv(output_path, index=False)
            print(f"Saved modified CSV to {output_path}")

if __name__ == '__main__':
    main()
