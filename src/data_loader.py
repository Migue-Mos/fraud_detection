import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.
    
    Args:
        path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    df = pd.read_csv(path)
    return df

def basic_info(df: pd.DataFrame):
    """
    Print basic information about the dataset.
    
    Args:
        df (pd.DataFrame): The dataset to inspect.
    """
    print("=== First 5 rows ===")
    print(df.head())
    print("\n=== Info ===")
    print(df.info())
    print("\n=== Description ===")
    print(df.describe())