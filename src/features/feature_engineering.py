import pandas as pd
import numpy as np
from typing import List, Dict, Any
import networkx as nx
from datetime import datetime, timedelta

def create_time_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from timestamp.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the transaction data with timestamp column.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with additional time-based features.
    """
    result_df = df.copy()
    
    if 'timestamp' not in result_df.columns:
        print("Warning: timestamp column not found. Skipping time-based features.")
        return result_df
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_dtype(result_df['timestamp']):
        result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
    
    # Extract time components
    result_df['hour'] = result_df['timestamp'].dt.hour
    result_df['day'] = result_df['timestamp'].dt.day
    result_df['day_of_week'] = result_df['timestamp'].dt.dayofweek
    result_df['is_weekend'] = result_df['day_of_week'].isin([5, 6]).astype(int)
    
    # Time periods
    result_df['time_period'] = pd.cut(
        result_df['hour'], 
        bins=[0, 6, 12, 18, 24], 
        labels=['night', 'morning', 'afternoon', 'evening'],
        include_lowest=True
    )
    
    # Convert to one-hot encoding
    time_period_dummies = pd.get_dummies(result_df['time_period'], prefix='time_period')
    result_df = pd.concat([result_df, time_period_dummies], axis=1)
    
    return result_df

def create_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features based on transaction characteristics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the transaction data.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with additional transaction-based features.
    """
    result_df = df.copy()
    
    # Gas efficiency
    if all(col in result_df.columns for col in ['gas_used', 'gas_limit']):
        result_df['gas_efficiency'] = result_df['gas_used'] / result_df['gas_limit']
    
    # Gas price relative to transaction value
    if all(col in result_df.columns for col in ['gas_price_gwei', 'value_eth']):
        # Avoid division by zero
        result_df['gas_price_to_value_ratio'] = np.where(
            result_df['value_eth'] > 0,
            result_df['gas_price_gwei'] / (result_df['value_eth'] * 1e9),  # Convert ETH to Gwei
            0
        )
    
    # Transaction complexity based on input data length
    if 'input_data_length' in result_df.columns:
        result_df['is_complex_transaction'] = (result_df['input_data_length'] > 100).astype(int)
    
    # Transaction value categories
    if 'value_eth' in result_df.columns:
        result_df['value_category'] = pd.cut(
            result_df['value_eth'],
            bins=[-0.001, 0.001, 0.1, 1, 10, float('inf')],
            labels=['zero', 'very_small', 'small', 'medium', 'large']
        )
        
        # Convert to one-hot encoding
        value_cat_dummies = pd.get_dummies(result_df['value_category'], prefix='value_cat')
        result_df = pd.concat([result_df, value_cat_dummies], axis=1)
    
    return result_df

def create_address_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features based on address behavior.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the transaction data.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with additional address-based features.
    """
    result_df = df.copy()
    
    # Count transactions per address
    if 'from_address' in result_df.columns:
        address_counts = result_df['from_address'].value_counts()
        result_df['from_address_tx_count'] = result_df['from_address'].map(address_counts)
    
    if 'to_address' in result_df.columns:
        address_counts = result_df['to_address'].value_counts()
        result_df['to_address_tx_count'] = result_df['to_address'].map(address_counts)
    
    # Unique addresses interacted with
    if all(col in result_df.columns for col in ['from_address', 'to_address']):
        # For each 'from_address', count unique 'to_address'
        from_to_unique = result_df.groupby('from_address')['to_address'].nunique()
        result_df['from_address_unique_interactions'] = result_df['from_address'].map(from_to_unique)
        
        # For each 'to_address', count unique 'from_address'
        to_from_unique = result_df.groupby('to_address')['from_address'].nunique()
        result_df['to_address_unique_interactions'] = result_df['to_address'].map(to_from_unique)
    
    # Self-transaction flag
    if all(col in result_df.columns for col in ['from_address', 'to_address']):
        result_df['is_self_transaction'] = (result_df['from_address'] == result_df['to_address']).astype(int)
    
    return result_df

def create_network_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create network-based features using graph analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the transaction data.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with additional network-based features.
    """
    result_df = df.copy()
    
    if not all(col in result_df.columns for col in ['from_address', 'to_address']):
        print("Warning: address columns not found. Skipping network features.")
        return result_df
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add edges from transactions
    for _, row in result_df.iterrows():
        from_addr = row['from_address']
        to_addr = row['to_address']
        
        if pd.notna(from_addr) and pd.notna(to_addr):
            if G.has_edge(from_addr, to_addr):
                # Increment weight if edge exists
                G[from_addr][to_addr]['weight'] += 1
            else:
                # Add new edge with weight 1
                G.add_edge(from_addr, to_addr, weight=1)
    
    # Calculate network metrics
    # In-degree (number of addresses that sent to this address)
    in_degree = dict(G.in_degree())
    # Out-degree (number of addresses this address sent to)
    out_degree = dict(G.out_degree())
    
    # Add metrics to DataFrame
    result_df['from_address_out_degree'] = result_df['from_address'].map(out_degree).fillna(0)
    result_df['to_address_in_degree'] = result_df['to_address'].map(in_degree).fillna(0)
    
    # Calculate PageRank (importance in the network)
    pagerank = nx.pagerank(G, alpha=0.85)
    result_df['from_address_pagerank'] = result_df['from_address'].map(pagerank).fillna(0)
    result_df['to_address_pagerank'] = result_df['to_address'].map(pagerank).fillna(0)
    
    return result_df

def create_temporal_features(df: pd.DataFrame, window_sizes: List[int] = [10, 50, 100]) -> pd.DataFrame:
    """
    Create temporal features using rolling windows.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the transaction data.
    window_sizes : List[int], default=[10, 50, 100]
        List of window sizes for rolling calculations.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with additional temporal features.
    """
    result_df = df.copy()
    
    if 'timestamp' not in result_df.columns:
        print("Warning: timestamp column not found. Skipping temporal features.")
        return result_df
    
    # Ensure timestamp is datetime and sort
    if not pd.api.types.is_datetime64_dtype(result_df['timestamp']):
        result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
    
    result_df = result_df.sort_values('timestamp')
    
    # Features to calculate rolling statistics for
    features = ['value_eth', 'gas_price_gwei', 'transaction_fee']
    
    for feature in features:
        if feature not in result_df.columns:
            continue
            
        for window in window_sizes:
            # Rolling mean
            result_df[f'{feature}_rolling_mean_{window}'] = result_df[feature].rolling(window=window).mean()
            
            # Rolling standard deviation
            result_df[f'{feature}_rolling_std_{window}'] = result_df[feature].rolling(window=window).std()
            
            # Rolling min/max
            result_df[f'{feature}_rolling_min_{window}'] = result_df[feature].rolling(window=window).min()
            result_df[f'{feature}_rolling_max_{window}'] = result_df[feature].rolling(window=window).max()
    
    # Fill NaN values created by rolling windows
    numeric_cols = result_df.select_dtypes(include=[np.number]).columns
    result_df[numeric_cols] = result_df[numeric_cols].fillna(0)
    
    return result_df

def select_features(df: pd.DataFrame, target_col: str = 'anomaly_flag') -> pd.DataFrame:
    """
    Select relevant features for model training.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing all features.
    target_col : str, default='anomaly_flag'
        Name of the target column.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with selected features.
    """
    # Columns to exclude from features
    exclude_cols = [
        'tx_hash',  # Unique identifier
        'timestamp',  # Raw timestamp
        'from_address', 'to_address',  # Raw addresses
        'input_data',  # Raw input data
        'transaction_status',  # Status as string
        'triangulation_group_id',  # Group ID
        target_col  # Target variable
    ]
    
    # Select features (all columns except excluded ones)
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Return features and target
    X = df[feature_cols]
    y = df[target_col] if target_col in df.columns else None
    
    return X, y