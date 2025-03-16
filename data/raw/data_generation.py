import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

# Fijamos semilla para reproducibilidad
np.random.seed(42)
random.seed(42)

# Función para generar strings hexadecimales (simula direcciones y hashes)
def random_hex_string(length):
    return "0x" + ''.join(random.choices('0123456789abcdef', k=length))

# Función para generar fechas aleatorias entre start y end
def random_date(start, end):
    delta = end - start
    random_seconds = random.randrange(int(delta.total_seconds()))
    return start + timedelta(seconds=random_seconds)

# Función para generar valores de gas
def generate_gas(anomalous, is_contract_interaction):
    if not anomalous:
        # Transacciones normales
        if is_contract_interaction == 1:
            gas_limit = np.random.randint(50000, 500000)
        else:
            gas_limit = np.random.randint(21000, 30000)
        gas_used = np.random.randint(21000, gas_limit + 1)
    else:
        # Transacciones anómalas: elegir entre dos escenarios
        if random.random() < 0.5:
            # Escenario de gas alto: límites muy altos y uso cercano al límite
            gas_limit = np.random.randint(6000000, 8000000)
            gas_used = np.random.randint(int(gas_limit * 0.9), gas_limit + 1)
        else:
            # Escenario de discrepancia: gas_limit en rango normal, pero gas_used muy bajo
            if is_contract_interaction == 1:
                gas_limit = np.random.randint(50000, 500000)
            else:
                gas_limit = np.random.randint(21000, 30000)
            gas_used = int(gas_limit * np.random.uniform(0.1, 0.3))
    return gas_limit, gas_used

# Parámetros del dataset
n_transactions = 100000
# Escalamos la cantidad de transacciones de triangulación proporcionalmente:
factor = n_transactions / 5000
n_triangulation_groups = int(50 * factor)  # 50 * 20 = 1000 grupos de triangulación
normal_transactions_count = n_transactions - (n_triangulation_groups * 3)

# Generamos un pool de billeteras (por ejemplo, 1000 billeteras únicas)
n_wallets = 1000
wallets = [random_hex_string(40) for _ in range(n_wallets)]

# Asignamos una probabilidad aleatoria de que una billetera sea sospechosa (este dato se usará en análisis posterior)
suspicious_wallet_ratio = np.random.uniform(0.05, 0.15)
wallet_flags = {wallet: (1 if random.random() < suspicious_wallet_ratio else 0) for wallet in wallets}

# Rango de fechas: desde el inicio de Ethereum hasta la fecha actual
start_date = datetime(2015, 7, 30)
end_date = datetime.now()

# --- Generación de transacciones normales ---
normal_data = {
    "tx_hash": [],
    "block_number": [],
    "timestamp": [],
    "from_address": [],
    "to_address": [],
    "value_eth": [],
    "gas_limit": [],
    "gas_used": [],
    "gas_price_gwei": [],
    "transaction_fee": [],
    "nonce": [],
    "tx_index": [],
    "is_contract_interaction": [],
    "input_data": [],
    "input_data_length": [],
    "transaction_status": [],
    "anomaly_flag": [],  # 0 para normal, 1 para anómala
    "transaction_type": []  # "transfer", "contract_interaction", "triangulation"
}

for _ in range(normal_transactions_count):
    from_wallet = random.choice(wallets)
    to_wallet = random.choice(wallets)
    while to_wallet == from_wallet:
        to_wallet = random.choice(wallets)
        
    tx_hash = random_hex_string(64)
    block_number = np.random.randint(1000000, 15000000)
    timestamp = random_date(start_date, end_date)
    value_eth = np.round(np.random.exponential(scale=1.0), 6)
    # Definimos si es interacción con contrato
    is_contract_interaction = np.random.choice([0, 1], p=[0.7, 0.3])
    # Asignamos el tipo de transacción en función de is_contract_interaction
    transaction_type = "contract_interaction" if is_contract_interaction == 1 else "transfer"
    # Para transacciones normales, el 5% se marcará como anómalo (por otros motivos)
    anomaly_flag = 1 if random.random() < 0.05 else 0
    # Generamos los valores de gas según si es anómalo o no
    gas_limit, gas_used = generate_gas(anomalous=(anomaly_flag==1), is_contract_interaction=is_contract_interaction)
    gas_price = np.random.randint(1, 200)
    transaction_fee = gas_limit * gas_price
    nonce = np.random.randint(0, 1000)
    tx_index = np.random.randint(0, 200)
    input_data = random_hex_string(random.randint(10, 100))
    input_data_length = len(input_data)
    transaction_status = "Success"  # Todas las transacciones son exitosas
    
    normal_data["tx_hash"].append(tx_hash)
    normal_data["block_number"].append(block_number)
    normal_data["timestamp"].append(timestamp)
    normal_data["from_address"].append(from_wallet)
    normal_data["to_address"].append(to_wallet)
    normal_data["value_eth"].append(value_eth)
    normal_data["gas_limit"].append(gas_limit)
    normal_data["gas_used"].append(gas_used)
    normal_data["gas_price_gwei"].append(gas_price)
    normal_data["transaction_fee"].append(transaction_fee)
    normal_data["nonce"].append(nonce)
    normal_data["tx_index"].append(tx_index)
    normal_data["is_contract_interaction"].append(is_contract_interaction)
    normal_data["input_data"].append(input_data)
    normal_data["input_data_length"].append(input_data_length)
    normal_data["transaction_status"].append(transaction_status)
    normal_data["anomaly_flag"].append(anomaly_flag)
    normal_data["transaction_type"].append(transaction_type)

normal_df = pd.DataFrame(normal_data)

# --- Generación de transacciones de triangulación ---
# Estas transacciones se consideran anómalas y se etiquetan con "triangulation"
triangulation_data = {
    "tx_hash": [],
    "block_number": [],
    "timestamp": [],
    "from_address": [],
    "to_address": [],
    "value_eth": [],
    "gas_limit": [],
    "gas_used": [],
    "gas_price_gwei": [],
    "transaction_fee": [],
    "nonce": [],
    "tx_index": [],
    "is_contract_interaction": [],
    "input_data": [],
    "input_data_length": [],
    "transaction_status": [],
    "anomaly_flag": [],
    "transaction_type": []
}

for group_id in range(1, n_triangulation_groups + 1):
    # Seleccionamos 3 billeteras distintas para formar el triángulo.
    group_wallets = random.sample(wallets, 3)
    # Si ninguna de las tres es sospechosa (según wallet_flags), forzamos que al menos la primera lo sea (aunque no almacenamos el flag)
    if not any(wallet_flags[w] for w in group_wallets):
        suspicious_pool = [w for w in wallets if wallet_flags[w] == 1]
        if suspicious_pool:
            group_wallets[0] = random.choice(suspicious_pool)
    # Definimos las transacciones del triángulo: A -> B, B -> C y C -> A.
    transactions = [
        (group_wallets[0], group_wallets[1]),
        (group_wallets[1], group_wallets[2]),
        (group_wallets[2], group_wallets[0])
    ]
    for from_wallet, to_wallet in transactions:
        tx_hash = random_hex_string(64)
        block_number = np.random.randint(1000000, 15000000)
        timestamp = random_date(start_date, end_date)
        value_eth = np.round(np.random.exponential(scale=1.0), 6)
        is_contract_interaction = np.random.choice([0, 1], p=[0.7, 0.3])
        # Para las transacciones de triangulación, se asigna siempre anomaly_flag = 1
        anomaly_flag = 1
        # El tipo de transacción será "triangulation"
        transaction_type = "triangulation"
        gas_limit, gas_used = generate_gas(anomalous=True, is_contract_interaction=is_contract_interaction)
        gas_price = np.random.randint(1, 200)
        transaction_fee = gas_limit * gas_price
        nonce = np.random.randint(0, 1000)
        tx_index = np.random.randint(0, 200)
        input_data = random_hex_string(random.randint(10, 100))
        input_data_length = len(input_data)
        transaction_status = "Success"
        
        triangulation_data["tx_hash"].append(tx_hash)
        triangulation_data["block_number"].append(block_number)
        triangulation_data["timestamp"].append(timestamp)
        triangulation_data["from_address"].append(from_wallet)
        triangulation_data["to_address"].append(to_wallet)
        triangulation_data["value_eth"].append(value_eth)
        triangulation_data["gas_limit"].append(gas_limit)
        triangulation_data["gas_used"].append(gas_used)
        triangulation_data["gas_price_gwei"].append(gas_price)
        triangulation_data["transaction_fee"].append(transaction_fee)
        triangulation_data["nonce"].append(nonce)
        triangulation_data["tx_index"].append(tx_index)
        triangulation_data["is_contract_interaction"].append(is_contract_interaction)
        triangulation_data["input_data"].append(input_data)
        triangulation_data["input_data_length"].append(input_data_length)
        triangulation_data["transaction_status"].append(transaction_status)
        triangulation_data["anomaly_flag"].append(anomaly_flag)
        triangulation_data["transaction_type"].append(transaction_type)

triangulation_df = pd.DataFrame(triangulation_data)

# Combinamos y mezclamos el dataset
df = pd.concat([normal_df, triangulation_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Visualizamos algunas filas y guardamos el dataset a CSV
print(df.head())
df.to_csv("synthetic_eth_transactions.csv", index=False)
