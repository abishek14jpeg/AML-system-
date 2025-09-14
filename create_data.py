# create_data.py (Updated Version)

import pandas as pd
import numpy as np
import os

def generate_data(num_rows=10000):
    """Generates a synthetic transaction dataset with 6 columns."""
    print(f"Generating {num_rows} synthetic transactions...")
    
    # Generate unique account IDs
    max_account_id = num_rows // 2
    sender_ids = np.random.randint(1000, max_account_id, size=num_rows)
    receiver_ids = np.random.randint(1000, max_account_id, size=num_rows)
    
    # Ensure sender and receiver are not the same
    same_id_indices = sender_ids == receiver_ids
    receiver_ids[same_id_indices] += 1

    data = {
        'amount': np.random.lognormal(mean=3.5, sigma=1.5, size=num_rows).round(2),
        'sender_account_age': np.random.randint(1, 3650, size=num_rows),
        'receiver_account_age': np.random.randint(1, 3650, size=num_rows),
        'is_fraud': np.random.choice([0, 1], size=num_rows, p=[0.98, 0.02]),
        'sender_account_id': sender_ids,
        'receiver_account_id': receiver_ids
    }
    
    df = pd.DataFrame(data)

    # Make fraudulent transactions look more suspicious
    fraud_indices = df[df['is_fraud'] == 1].index
    df.loc[fraud_indices, 'amount'] *= np.random.uniform(5, 20, size=len(fraud_indices))
    df.loc[fraud_indices, 'sender_account_age'] = np.random.randint(1, 30, size=len(fraud_indices))
    
    # Define the output path
    output_path = os.path.join('data', 'synthetic_data.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Reorder columns to a logical format
    df = df[['amount', 'sender_account_age', 'receiver_account_age', 'is_fraud', 'sender_account_id', 'receiver_account_id']]
    
    df.to_csv(output_path, index=False)
    print(f"Successfully created '{output_path}' with 6 columns.")

if __name__ == '__main__':
    generate_data()