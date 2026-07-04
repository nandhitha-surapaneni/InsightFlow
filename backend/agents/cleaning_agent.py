import pandas as pd


def clean_data(df: pd.DataFrame):
    total_rows = len(df)
    total_cols = len(df.columns)

    # Missing values
    missing_values = int(df.isnull().sum().sum())

    # Missing values per column
    missing_by_column = {
        col: int(df[col].isnull().sum())
        for col in df.columns
        if df[col].isnull().sum() > 0
    }

    # Duplicate rows
    duplicate_rows = int(df.duplicated().sum())

    # Column type detection
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_columns = df.select_dtypes(include=["datetime"]).columns.tolist()

    column_types = {}
    for col in df.columns:
        if col in numeric_columns:
            column_types[col] = "numeric"
        elif col in categorical_columns:
            column_types[col] = "categorical"
        elif col in datetime_columns:
            column_types[col] = "datetime"
        else:
            column_types[col] = "other"

    # Quality score
    total_cells = total_rows * total_cols if total_rows and total_cols else 1
    missing_ratio = missing_values / total_cells
    duplicate_ratio = duplicate_rows / total_rows if total_rows else 0

    quality_score = max(0, round(100 - ((missing_ratio * 70) + (duplicate_ratio * 30)) * 100, 2))

    # cleaning log
    cleaning_log = []
    if missing_values > 0:
      cleaning_log.append(
        f"Detected {missing_values} missing values across dataset."
     )

    if duplicate_rows > 0:
      cleaning_log.append(
        f"Detected {duplicate_rows} duplicate rows."
     )

    if missing_values == 0 and duplicate_rows == 0:
      cleaning_log.append(
        "Dataset passed quality checks with no missing or duplicate values."
     )

    return {
        "totalRows": total_rows,
        "totalColumns": total_cols,
        "missingValues": missing_values,
        "missingByColumn": missing_by_column,
        "duplicateRows": duplicate_rows,
        "columnTypes": column_types,
        "numericColumns": numeric_columns,
        "categoricalColumns": categorical_columns,
        "datetimeColumns": datetime_columns,
        "qualityScore": quality_score,
        "cleaningLog": cleaning_log
    }