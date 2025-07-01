import pandas as pd
import io

def predict_student(data, model_name):
    """
    Placeholder for student prediction ML pipeline.
    Simulates a prediction based on input data and model name.
    """
    print(f"ML: Predicting for student with data: {data} using model: {model_name}")
    # Simulate different outcomes based on model or data for variety
    if model_name == 'logistic':
        probability = 0.65 + (data.get('motivation', 3) - 3) * 0.05 - (data.get('absences', 0) * 0.01)
    elif model_name == 'tree':
        probability = 0.75 + (data.get('study_time', 0) / 20) - (data.get('age', 18) - 18) * 0.02
    else: # rf or other
        probability = 0.85 - (data.get('absences', 0) * 0.02) + (data.get('previous_grade', 50) / 1000)

    probability = max(0, min(1, probability)) # Ensure probability is between 0 and 1
    passed = probability >= 0.5

    return {
        "passed": passed,
        "probability_score": round(probability, 3),
        "model_used": model_name
    }

def batch_predict(file_path_or_buffer):
    """
    Placeholder for batch prediction ML pipeline.
    Simulates reading a CSV, making predictions, and returning results.
    """
    try:
        if isinstance(file_path_or_buffer, str):
            df = pd.read_csv(file_path_or_buffer)
        else: # assume it's a file-like object (UploadedFile)
            file_path_or_buffer.seek(0) # Go to the start of the file
            df = pd.read_csv(io.StringIO(file_path_or_buffer.read().decode('utf-8')))
    except Exception as e:
        print(f"Error reading CSV for batch prediction: {e}")
        # Return an empty DataFrame or raise an error, depending on desired handling
        return pd.DataFrame()


    print(f"ML: Processing batch prediction for {len(df)} records.")

    # Simulate predictions - in a real scenario, you'd apply your model(s) here
    # For this placeholder, we'll add some dummy prediction columns.
    # We'll use a simplified logic based on 'Previous Grade' and 'Absences' if they exist.

    required_columns = ['Previous Grade', 'Absences', 'Study Time', 'Age', 'Internet Access', 'Extra Courses', 'Motivation', 'Parent Education']

    # Check if required columns exist, if not, use default values for prediction logic
    for col in required_columns:
        if col not in df.columns:
            # For simplicity, let's assume a default value if a column is missing.
            # A more robust solution would involve error handling or specific imputation.
            if col in ['Previous Grade', 'Study Time', 'Age', 'Motivation', 'Parent Education']:
                 df[col] = 50 if col == 'Previous Grade' else (2 if col == 'Study Time' else (18 if col == 'Age' else 3))
            elif col == 'Absences':
                df[col] = 0
            else: # Boolean columns
                df[col] = False


    df['Predicted_Passed_Logistic'] = (df['Previous Grade'] > 60) & (df['Absences'] < 10)
    df['Predicted_Score_Logistic'] = df.apply(
        lambda row: 0.7 if (row['Previous Grade'] > 60 and row['Absences'] < 10) else 0.3, axis=1
    )

    df['Predicted_Passed_Tree'] = (df['Study Time'] > 10) & (df['Motivation'] > 3)
    df['Predicted_Score_Tree'] = df.apply(
        lambda row: 0.8 if (row['Study Time'] > 10 and row['Motivation'] > 3) else 0.4, axis=1
    )

    # For simplicity, let's say the researcher flow always uses a "default_batch_model"
    # or you could make it choose one. Here we just add some dummy columns.
    df['prediction_passed'] = (df['Previous Grade'] > 50) & (df['Absences'] < 5)
    df['prediction_probability'] = df.apply(
        lambda row: 0.75 if (row['Previous Grade'] > 50 and row['Absences'] < 5) else 0.4, axis=1
    )

    # Keep original columns + new prediction columns
    # The view will select which columns to display from this.
    return df

def get_sample_csv_data():
    """
    Returns a string representing sample CSV data.
    """
    sample_data = {
        'Previous Grade': [75, 88, 60, 92, 78],
        'Absences': [2, 0, 5, 1, 3],
        'Study Time': [10, 15, 8, 20, 12],
        'Age': [17, 18, 17, 19, 18],
        'Internet Access': [True, True, False, True, True],
        'Extra Courses': [False, True, False, True, False],
        'Motivation': [4, 5, 3, 5, 4],
        'Parent Education': [3, 4, 2, 5, 3]
    }
    df = pd.DataFrame(sample_data)
    return df.to_csv(index=False)
