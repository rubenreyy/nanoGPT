import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os

# Define the columns for the KDD Cup dataset
columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack", "last_flag"
]

def preprocess_data(data, encoder=None, train_encoded_shape=None):
    # Preprocess categorical features
    nominal_features = ['protocol_type', 'service', 'flag']
    
    # Fit the encoder only on training data, but transform both train and test
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_features = encoder.fit_transform(data[nominal_features])
        train_encoded_shape = encoded_features.shape  # Capture the shape of training data encoded features
    else:
        encoded_features = encoder.transform(data[nominal_features])
        
        # Ensure test data has the same number of encoded columns as train data
        if encoded_features.shape[1] < train_encoded_shape[1]:
            # Add zero columns for missing categories in test data
            missing_cols = train_encoded_shape[1] - encoded_features.shape[1]
            encoded_features = np.hstack([encoded_features, np.zeros((encoded_features.shape[0], missing_cols))])

    # Preprocess numerical features
    scaler = StandardScaler()
    numerical_features = [
        'duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent',
        # (other numerical features here)
    ]
    scaled_numerical = scaler.fit_transform(data[numerical_features])
    
    # Combine features
    processed_data = np.hstack((encoded_features, scaled_numerical))
    
    # Define target variable for binary classification
    data['target'] = data['attack'].apply(lambda x: 0 if x == 'normal' else 1)
    
    # Convert to PyTorch tensors
    X = torch.tensor(processed_data, dtype=torch.float32)
    y = torch.tensor(data['target'].values, dtype=torch.float32)
    
    return X, y, encoder, train_encoded_shape

if __name__ == "__main__":
    train_filepath = r'c:\Users\Ruben\Networks\NanoGPT Toy experiment\nanoGPT\data\shakespeare_char\Train.txt'
    test_filepath = r'c:\Users\Ruben\Networks\NanoGPT Toy experiment\nanoGPT\data\shakespeare_char\Test.txt'

    if os.path.exists(train_filepath) and os.path.exists(test_filepath):
        print("Both train and test files found.")
        
        # Preprocess train data and save encoder
        train_data = pd.read_csv(train_filepath, names=columns)
        X_train, y_train, encoder, train_encoded_shape = preprocess_data(train_data)

        # Preprocess test data using the same encoder and ensure shape consistency
        test_data = pd.read_csv(test_filepath, names=columns)
        X_test, y_test, _, _ = preprocess_data(test_data, encoder, train_encoded_shape)

        # Save the processed data as PyTorch tensors
        torch.save(X_train, 'X_train.pt')
        torch.save(y_train, 'y_train.pt')
        torch.save(X_test, 'X_test.pt')
        torch.save(y_test, 'y_test.pt')

        print("Processed data saved as .pt files.")
    else:
        print("One or both files not found.")
