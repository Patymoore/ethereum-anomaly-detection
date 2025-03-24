# Ethereum Transaction Anomaly Detection with Triangulation Analysis

This project implements a comprehensive system for detecting anomalous transactions in the Ethereum blockchain, with a specific focus on triangulation patterns. The system analyzes transaction patterns, identifies suspicious behaviors, and flags potentially fraudulent activities.

## Project Structure

```
ethereum-anomaly-detection/
│
├── data/
│   ├── raw/                # Raw dataset
│   └── processed/          # Processed datasets
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_triangulation_analysis.ipynb  # Triangulation analysis
│   ├── 04_model_development.ipynb
│   └── 05_results_visualization.ipynb
│
│
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ethereum-anomaly-detection.git
   cd ethereum-anomaly-detection
   ```

2. **Create a virtual environment and activate it:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The project uses a synthetic Ethereum transaction dataset with the following features:

- Transaction details (hash, block number, timestamp)
- Address information (from_address, to_address)
- Value and gas information (value_eth, gas_limit, gas_used, gas_price_gwei, transaction_fee)
- Transaction metadata (nonce, tx_index, is_contract_interaction, input_data)
- Anomaly indicators (anomaly_flag, transaction_type)

## Usage

### Data Preprocessing and Exploration
Run the data exploration notebook to understand the dataset and preprocess it:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### Feature Engineering
Run the feature engineering notebook to create relevant features for anomaly detection:

```bash
jupyter notebook notebooks/02_feature_engineering.ipynb
```

### Triangulation Analysis
Run the triangulation analysis notebook to identify and analyze triangulation patterns:

```bash
jupyter notebook notebooks/03_triangulation_analysis.ipynb
```

### Model Development
Run the model development notebook to train and evaluate anomaly detection models:

```bash
jupyter notebook notebooks/04_model_development.ipynb
```

### Results Visualization
Run the results visualization notebook to visualize the model results:

```bash
jupyter notebook notebooks/05_results_visualization.ipynb
```

## Methodology

### 1. Data Preprocessing
- Handling missing values
- Converting data types
- Handling outliers
- Creating derived features

### 2. Feature Engineering
- Transaction-based features
- Address-based features
- Temporal features
- Network-based features

### 3. Triangulation Analysis
- Identifying cycles in transaction networks
- Analyzing temporal patterns in triangulation
- Comparing triangulation vs. normal transactions
- Network analysis of triangulation patterns

### 4. Anomaly Detection Models
- Isolation Forest
- One-Class SVM
- Local Outlier Factor
- Autoencoder
- XGBoost (supervised approach)

### 5. Evaluation
- Precision, Recall, F1-score
- ROC-AUC
- Confusion Matrix
- Feature Importance Analysis

## Triangulation Detection

Triangulation refers to a pattern where funds move in a cycle between three or more addresses, potentially indicating money laundering or other suspicious activities. This project implements several techniques to detect and analyze triangulation patterns:

- **Cycle Detection**: Using graph theory to identify cycles in the transaction network.
- **Temporal Analysis**: Analyzing the time span and sequence of transactions within triangulation patterns.
- **Value Analysis**: Examining the value distribution and consistency within triangulation cycles.
- **Network Analysis**: Identifying key addresses and communities involved in triangulation activities.

## Results

The project successfully identifies anomalous transactions with high precision and recall. The most effective features for detecting anomalies include:

- Unusual transaction values
- Abnormal gas usage patterns
- Suspicious address interactions
- Temporal anomalies
- Triangulation patterns

## Future Work

- Implement real-time anomaly detection
- Incorporate additional blockchain data sources
- Develop more sophisticated network analysis techniques
- Explore deep learning approaches for sequence modeling
- Extend triangulation detection to larger cycles and more complex patterns

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project uses synthetic data for educational purposes
- Thanks to the open-source community for providing the tools and libraries used in this project

## Step-by-Step Guide to Run the Project

### 1. Set Up the Environment
```bash
# Clone the repository (or create the directory structure manually)
git clone https://github.com/patymoore/ethereum-anomaly-detection.git
cd ethereum-anomaly-detection

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt