import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
import networkx as nx

def plot_feature_distributions(df: pd.DataFrame, features: List[str], 
                              hue: Optional[str] = None) -> None:
    """
    Plot distributions of selected features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data.
    features : List[str]
        List of feature names to plot.
    hue : str, optional
        Column name to use for color encoding.
    """
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        if i < len(axes):
            if feature in df.columns:
                if df[feature].dtype in [np.float64, np.int64]:
                    sns.histplot(data=df, x=feature, hue=hue, kde=True, ax=axes[i])
                else:
                    sns.countplot(data=df, x=feature, hue=hue, ax=axes[i])
                
                axes[i].set_title(f'Distribution of {feature}')
                axes[i].tick_params(axis='x', rotation=45)
            else:
                axes[i].set_visible(False)
    
    # Hide any unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df: pd.DataFrame, features: List[str], 
                           title: str = 'Correlation Matrix') -> None:
    """
    Plot correlation matrix for selected features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data.
    features : List[str]
        List of feature names to include in the correlation matrix.
    title : str, default='Correlation Matrix'
        Title for the plot.
    """
    # Select only features that exist in the DataFrame
    valid_features = [f for f in features if f in df.columns]
    
    # Calculate correlation matrix
    corr_matrix = df[valid_features].corr()
    
    # Plot
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=True, 
                fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_time_series(df: pd.DataFrame, time_col: str, value_col: str, 
                    group_col: Optional[str] = None, 
                    title: str = 'Time Series Plot') -> None:
    """
    Plot time series data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data.
    time_col : str
        Column name for time values.
    value_col : str
        Column name for the values to plot.
    group_col : str, optional
        Column name for grouping the data.
    title : str, default='Time Series Plot'
        Title for the plot.
    """
    plt.figure(figsize=(14, 6))
    
    if group_col and group_col in df.columns:
        for group, group_df in df.groupby(group_col):
            plt.plot(group_df[time_col], group_df[value_col], label=str(group))
        plt.legend(title=group_col)
    else:
        plt.plot(df[time_col], df[value_col])
    
    plt.title(title)
    plt.xlabel(time_col)
    plt.ylabel(value_col)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_scatter_matrix(df: pd.DataFrame, features: List[str], 
                       hue: Optional[str] = None) -> None:
    """
    Plot scatter matrix for selected features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data.
    features : List[str]
        List of feature names to include in the scatter matrix.
    hue : str, optional
        Column name to use for color encoding.
    """
    # Select only features that exist in the DataFrame
    valid_features = [f for f in features if f in df.columns]
    
    # Create scatter matrix
    sns.pairplot(df[valid_features + ([hue] if hue else [])], 
                hue=hue, diag_kind='kde', plot_kws={'alpha': 0.6})
    plt.suptitle('Scatter Matrix', y=1.02)
    plt.tight_layout()
    plt.show()

def plot_transaction_network(df: pd.DataFrame, from_col: str = 'from_address', 
                            to_col: str = 'to_address', value_col: Optional[str] = 'value_eth',
                            anomaly_col: Optional[str] = 'anomaly_flag',
                            max_nodes: int = 50) -> None:
    """
    Plot transaction network graph.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the transaction data.
    from_col : str, default='from_address'
        Column name for source addresses.
    to_col : str, default='to_address'
        Column name for destination addresses.
    value_col : str, optional
        Column name for transaction values.
    anomaly_col : str, optional
        Column name for anomaly flag.
    max_nodes : int, default=50
        Maximum number of nodes to display.
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Limit to most active addresses to avoid cluttering
    if len(df) > max_nodes:
        # Count transactions per address
        from_counts = df[from_col].value_counts()
        to_counts = df[to_col].value_counts()
        
        # Combine counts
        all_counts = pd.concat([from_counts, to_counts]).groupby(level=0).sum()
        
        # Get top addresses
        top_addresses = all_counts.nlargest(max_nodes).index
        
        # Filter transactions involving top addresses
        filtered_df = df[
            (df[from_col].isin(top_addresses)) | 
            (df[to_col].isin(top_addresses))
        ]
    else:
        filtered_df = df
    
    # Add edges from transactions
    for _, row in filtered_df.iterrows():
        from_addr = row[from_col]
        to_addr = row[to_col]
        
        # Skip self-transactions for clarity
        if from_addr == to_addr:
            continue
        
        # Add edge with attributes
        if G.has_edge(from_addr, to_addr):
            # Increment weight if edge exists
            G[from_addr][to_addr]['weight'] += 1
            if value_col in row and pd.notna(row[value_col]):
                G[from_addr][to_addr]['value'] += row[value_col]
            
            # Mark as anomalous if any transaction is anomalous
            if anomaly_col in row and row[anomaly_col]:
                G[from_addr][to_addr]['anomalous'] = True
        else:
            # Add new edge
            edge_attrs = {'weight': 1}
            if value_col in row and pd.notna(row[value_col]):
                edge_attrs['value'] = row[value_col]
            if anomaly_col in row:
                edge_attrs['anomalous'] = bool(row[anomaly_col])
            
            G.add_edge(from_addr, to_addr, **edge_attrs)
    
    # Calculate node positions using spring layout
    pos = nx.spring_layout(G, k=0.3, iterations=50)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, 
        node_size=[G.degree(node) * 20 for node in G.nodes()],
        node_color='skyblue',
        alpha=0.8
    )
    
    # Draw edges with different colors for anomalous transactions
    normal_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get('anomalous', False)]
    anomalous_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('anomalous', False)]
    
    # Draw normal edges
    nx.draw_networkx_edges(
        G, pos, 
        edgelist=normal_edges,
        width=[G[u][v]['weight'] * 0.5 for u, v in normal_edges],
        alpha=0.5,
        edge_color='gray',
        arrows=True,
        arrowsize=10
    )
    
    # Draw anomalous edges
    nx.draw_networkx_edges(
        G, pos, 
        edgelist=anomalous_edges,
        width=[G[u][v]['weight'] * 0.5 for u, v in anomalous_edges],
        alpha=0.7,
        edge_color='red',
        arrows=True,
        arrowsize=15
    )
    
    # Add labels to nodes (shortened for readability)
    labels = {node: node[:6] + '...' for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    
    plt.title('Transaction Network Graph')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_interactive_scatter(df: pd.DataFrame, x: str, y: str, 
                            color: Optional[str] = None, 
                            size: Optional[str] = None,
                            hover_data: List[str] = None) -> go.Figure:
    """
    Create interactive scatter plot using Plotly.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data.
    x : str
        Column name for x-axis.
    y : str
        Column name for y-axis.
    color : str, optional
        Column name for color encoding.
    size : str, optional
        Column name for point size.
    hover_data : List[str], optional
        List of column names to show in hover tooltip.
        
    Returns:
    --------
    go.Figure
        Plotly figure object.
    """
    fig = px.scatter(
        df, x=x, y=y, 
        color=color,
        size=size,
        hover_data=hover_data,
        title=f'{y} vs {x}',
        labels={
            x: x.replace('_', ' ').title(),
            y: y.replace('_', ' ').title()
        }
    )
    
    fig.update_layout(
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return fig