import pandas as pd
import os
from typing import Optional, Tuple

def load_ethereum_data(file_path: str) -> pd.DataFrame:
    """
    Load Ethereum transaction data from CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing Ethereum transaction data.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the Ethereum transaction data.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Print basic information about the loaded data
        print(f"Loaded {len(df)} transactions from {file_path}")
        print(f"Columns: {df.columns.tolist()}")
        
        return df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def split_data(df: pd.DataFrame, test_size: float = 0.2, 
               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into training and testing sets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data to split.
    test_size : float, default=0.2
        Proportion of the data to include in the test split.
    random_state : int, default=42
        Random seed for reproducibility.
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        Training and testing DataFrames.
    """
    from sklearn.model_selection import train_test_split
    
    # Shuffle the data and split
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set: {len(train_df)} transactions")
    print(f"Testing set: {len(test_df)} transactions")
    
    return train_df, test_df

def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save processed data to CSV file.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the processed data.
    output_path : str
        Path where the processed data will be saved.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")