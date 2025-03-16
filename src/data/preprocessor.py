import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Ethereum transaction data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the raw Ethereum transaction data.
        
    Returns:
    --------
    pd.DataFrame
        Cleaned DataFrame.
    """
    # Create a copy to avoid modifying the original DataFrame
    cleaned_df = df.copy()
    
    # Convert timestamp to datetime
    if 'timestamp' in cleaned_df.columns:
        cleaned_df['timestamp'] = pd.to_datetime(cleaned_df['timestamp'])
    
    # Convert numeric columns
    numeric_columns = [
        'block_number', 'value_eth', 'gas_limit', 'gas_used', 
        'gas_price_gwei', 'transaction_fee', 'nonce', 'tx_index',
        'is_contract_interaction', 'input_data_length',
        'from_suspicious', 'to_suspicious', 'anomaly_flag'
    ]
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
    
    # Handle missing values
    for col in cleaned_df.columns:
        missing_count = cleaned_df[col].isna().sum()
        if missing_count > 0:
            print(f"Column {col} has {missing_count} missing values")
            
            if col in numeric_columns:
                # Fill numeric columns with median
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            else:
                # Fill categorical columns with mode
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])
    
    # Convert boolean columns
    bool_columns = ['is_contract_interaction', 'from_suspicious', 'to_suspicious', 'anomaly_flag']
    for col in bool_columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].astype(bool)
    
    # Add derived timestamp columns
    if 'timestamp' in cleaned_df.columns:
        cleaned_df['hour_of_day'] = cleaned_df['timestamp'].dt.hour
        cleaned_df['day_of_week'] = cleaned_df['timestamp'].dt.dayofweek
        cleaned_df['month'] = cleaned_df['timestamp'].dt.month
        cleaned_df['year'] = cleaned_df['timestamp'].dt.year
        cleaned_df['date'] = cleaned_df['timestamp'].dt.date
    
    # Create gas efficiency ratio
    if all(col in cleaned_df.columns for col in ['gas_used', 'gas_limit']):
        cleaned_df['gas_efficiency'] = cleaned_df['gas_used'] / cleaned_df['gas_limit']
    
    # Create fee to value ratio (avoiding division by zero)
    if all(col in cleaned_df.columns for col in ['transaction_fee', 'value_eth']):
        cleaned_df['fee_to_value_ratio'] = np.where(
            cleaned_df['value_eth'] > 0,
            cleaned_df['transaction_fee'] / cleaned_df['value_eth'],
            0
        )
    
    return cleaned_df

def handle_outliers(df: pd.DataFrame, columns: List[str], method: str = 'clip') -> pd.DataFrame:
    """
    Handle outliers in specified columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data.
    columns : List[str]
        List of column names to check for outliers.
    method : str, default='clip'
        Method to handle outliers ('clip' or 'remove').
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with outliers handled.
    """
    result_df = df.copy()
    
    for col in columns:
        if col not in result_df.columns or not pd.api.types.is_numeric_dtype(result_df[col]):
            continue
            
        Q1 = result_df[col].quantile(0.25)
        Q3 = result_df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers
        outliers = ((result_df[col] < lower_bound) | (result_df[col] > upper_bound)).sum()
        print(f"Column {col}: {outliers} outliers detected")
        
        if method == 'clip':
            # Clip values to the bounds
            result_df[col] = result_df[col].clip(lower_bound, upper_bound)
        elif method == 'remove':
            # Remove rows with outliers
            result_df = result_df[
                (result_df[col] >= lower_bound) & 
                (result_df[col] <= upper_bound)
            ]
    
    return result_df

def normalize_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Normalize specified numeric features to [0, 1] range.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data.
    columns : List[str]
        List of column names to normalize.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with normalized features.
    """
    from sklearn.preprocessing import MinMaxScaler
    
    result_df = df.copy()
    scaler = MinMaxScaler()
    
    # Select only numeric columns that exist in the DataFrame
    valid_columns = [col for col in columns if col in result_df.columns 
                    and pd.api.types.is_numeric_dtype(result_df[col])]
    
    if valid_columns:
        result_df[valid_columns] = scaler.fit_transform(result_df[valid_columns])
    
    return result_df

def encode_categorical_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Encode categorical features using one-hot encoding.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data.
    columns : List[str]
        List of categorical column names to encode.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with encoded categorical features.
    """
    result_df = df.copy()
    
    for col in columns:
        if col in result_df.columns and not pd.api.types.is_numeric_dtype(result_df[col]):
            # Apply one-hot encoding
            dummies = pd.get_dummies(result_df[col], prefix=col, drop_first=True)
            result_df = pd.concat([result_df, dummies], axis=1)
            result_df = result_df.drop(col, axis=1)
    
    return result_df

def segment_by_transaction_type(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Segment the data by transaction type.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the transaction data.
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary with transaction types as keys and corresponding DataFrames as values.
    """
    if 'transaction_type' not in df.columns:
        print("Warning: transaction_type column not found. Cannot segment data.")
        return {'all': df}
    
    # Get unique transaction types
    transaction_types = df['transaction_type'].unique()
    
    # Create a dictionary to store segmented data
    segments = {}
    
    # Segment data by transaction type
    for tx_type in transaction_types:
        segments[tx_type] = df[df['transaction_type'] == tx_type].copy()
        print(f"Transaction type '{tx_type}': {len(segments[tx_type])} transactions")
    
    # Also include the full dataset
    segments['all'] = df
    
    return segments