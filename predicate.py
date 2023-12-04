import pandas as pd


def predicate_intro_outro_duration(file_path, new_total_duration):
    # Read the Excel file
    df = pd.read_excel(file_path)

    # Extract the relevant columns
    intro_duration = df['片头时长']
    outro_duration = df['片尾时长']
    total_duration = df['时长(秒)']

    # Calculate the correlation coefficients
    intro_total_corr = intro_duration.corr(total_duration)
    outro_total_corr = outro_duration.corr(total_duration)

    # Print the correlation coefficients
    print("Correlation between intro duration and total duration:", intro_total_corr)
    print("Correlation between outro duration and total duration:", outro_total_corr)

    # Predict new intro and outro durations based on total duration
    new_intro_duration = intro_duration.mean() + intro_total_corr * (new_total_duration - total_duration.mean()) / total_duration.std()
    new_outro_duration = outro_duration.mean() + outro_total_corr * (new_total_duration - total_duration.mean()) / total_duration.std()

    # Print the predicted intro and outro durations
    print("Predicted intro duration for a total duration of", new_total_duration, "seconds:", new_intro_duration)
    print("Predicted outro duration for a total duration of", new_total_duration, "seconds:", new_outro_duration)


# Call the function with the file path and desired total duration
file_path = '少儿动漫152.xlsx'
new_total_duration = 1000  # Replace with the desired total duration in seconds
predicate_intro_outro_duration(file_path, new_total_duration)


def calculate_intro_outro_stats(file_path):
    # Read the Excel file
    df = pd.read_excel(file_path)

    # Extract the relevant columns
    intro_duration = df['片头时长']
    outro_duration = df['片尾时长']
    total_duration = df['时长(秒)']

    # Calculate the correlation coefficients
    intro_total_corr = intro_duration.corr(total_duration)
    outro_total_corr = outro_duration.corr(total_duration)

    # Calculate the mean and standard deviation
    intro_duration_mean = intro_duration.mean()
    outro_duration_mean = outro_duration.mean()
    total_duration_mean = total_duration.mean()
    total_duration_std = total_duration.std()

    # Create a dictionary with the calculated values
    result = {
        'intro_correlation_coefficient': intro_total_corr,
        'outro_correlation_coefficient': outro_total_corr,
        'intro_duration_mean': intro_duration_mean,
        'outro_duration_mean': outro_duration_mean,
        'total_duration_mean': total_duration_mean,
        'total_duration_std': total_duration_std
    }

    return result


# Call the function with the file path
file_path = '少儿动漫152.xlsx'
output_dict = calculate_intro_outro_stats(file_path)
print(output_dict)
