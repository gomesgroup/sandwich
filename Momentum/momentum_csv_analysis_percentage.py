import pandas as pd
import re
import os

def extract_step_count(filename):
    match = re.search(r'_(\d+)steps', filename)
    return int(match.group(1)) if match else None

def analyze_directory(directory):
    summary_data = []
    step_columns = [f"{i} steps" for i in range(10, 101, 5)]
    total_files = 0

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            total_files += 1
            file_path = os.path.join(directory, filename)
            try:
                df = pd.read_csv(file_path)
                step_count = extract_step_count(filename)
                first_yes_count = df["Yes Count"][0]
                max_yes = int(re.search(r"max yes=(\d+)", first_yes_count).group(1))
                start_step = int(re.search(r"start=(\d+)", first_yes_count).group(1))
                end_step = int(re.search(r"end=(\d+)", first_yes_count).group(1))

                # Evaluate DM column based on criteria
                if abs(step_count - end_step) <= 100:
                    steps_range = df.loc[end_step:step_count, "Bonds > 1.8"]
                    if not steps_range.isnull().any() and (steps_range.diff().fillna(0) >= 0).all():
                        dm_status = "yes"
                    else:
                        dm_status = "no"
                else:
                    dm_status = "no"

                step_results = {}
                for step_col in step_columns:
                    required_yes = int(step_col.split()[0])
                    if max_yes >= required_yes:
                        if dm_status == "yes":
                            step_results[step_col] = "1"
                        else:
                            step_results[step_col] = "0"
                    else:
                        step_results[step_col] = "n/a"

                # Check if the value at start_step is 0
                start_step_row = df[df["Step"] == start_step]
                start_step_value = start_step_row["Bonds > 1.8"].values[0] if not start_step_row.empty else None
                if start_step_value != 0:
                    for step_col in step_columns:
                        step_results[step_col] = "---"

                end_step_row = df[df["Step"] == end_step]
                bonds_value = end_step_row["Bonds > 1.8"].values[0] if not end_step_row.empty else None

                # Check if bonds_value is 1 or 0, and set all step_results to "n/a"
                if bonds_value in [2, 1, 0]:
                    for step_col in step_columns:
                        step_results[step_col] = "n/a"

                row_data = {
                    "file name": filename,
                    "steps": step_count,
                    "momentum": first_yes_count,
                    "DM": dm_status,
                    "Bonds > 1.8": bonds_value,
                    **step_results
                }
                summary_data.append(row_data)
            except Exception as e:
                print(f"Error processing file: {filename}")
                print(f"Error message: {str(e)}")

    df_summary = pd.DataFrame(summary_data)

    # Add summary rows for each step column
    for step_col in step_columns:
        ones = df_summary[step_col].value_counts().get("1", 0)
        zeros = df_summary[step_col].value_counts().get("0", 0)
        if total_files != 0:
            ratio = f"{ones}/{total_files}"
            percent = round(ones / total_files * 100, 2)
            summary_value = f"{ratio} ({percent}%)"
        else:
            summary_value = "undefined"
        df_summary.loc["Summary", step_col] = summary_value

    # Save the new CSV file
    output_path = os.path.join(directory, "results.csv")
    df_summary.to_csv(output_path, index=False)
    print(f"Summary CSV created at {output_path}")

# Usage
directory_path = '/path'
analyze_directory(directory_path)
