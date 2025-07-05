import pandas as pd
import os
import hashlib

def calculate_file_hash(filepath):
    """Calculates the SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(8192)  # Read in 8KB chunks
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()

def load_processed_files_db(data_storage_root_dir):
    """Loads the processed files DataFrame from a Parquet file within the specified directory."""
    db_path = os.path.join(data_storage_root_dir, 'processed_files.parquet')
    if os.path.exists(db_path):
        return pd.read_parquet(db_path)
    return pd.DataFrame(columns=['filepath', 'hash'])

def save_processed_files_db(df, data_storage_root_dir):
    """Saves the processed files DataFrame to a Parquet file within the specified directory."""
    os.makedirs(data_storage_root_dir, exist_ok=True)
    db_path = os.path.join(data_storage_root_dir, 'processed_files.parquet')
    df.to_parquet(db_path, index=False)

def load_entities_db(data_storage_root_dir):
    """Loads the extracted entities DataFrame from a Parquet file."""
    db_path = os.path.join(data_storage_root_dir, 'entities.parquet')
    if os.path.exists(db_path):
        return pd.read_parquet(db_path)
    return pd.DataFrame(columns=['name', 'type'])

def save_entities_db(df, data_storage_root_dir):
    """Saves the extracted entities DataFrame to a Parquet file."""
    os.makedirs(data_storage_root_dir, exist_ok=True)
    db_path = os.path.join(data_storage_root_dir, 'entities.parquet')
    df.to_parquet(db_path, index=False)

def load_relationships_db(data_storage_root_dir):
    """Loads the extracted relationships DataFrame from a Parquet file."""
    db_path = os.path.join(data_storage_root_dir, 'relationships.parquet')
    if os.path.exists(db_path):
        return pd.read_parquet(db_path)
    return pd.DataFrame(columns=['source', 'target', 'type', 'description'])

def save_relationships_db(df, data_storage_root_dir):
    """Saves the extracted relationships DataFrame to a Parquet file."""
    os.makedirs(data_storage_root_dir, exist_ok=True)
    db_path = os.path.join(data_storage_root_dir, 'relationships.parquet')
    df.to_parquet(db_path, index=False)

def load_community_db(data_storage_root_dir):
    """Loads the detected communities DataFrame from a Parquet file."""
    db_path = os.path.join(data_storage_root_dir, 'communities.parquet')
    if os.path.exists(db_path):
        return pd.read_parquet(db_path)
    return pd.DataFrame(columns=['level', 'cluster_id', 'parent_cluster', 'nodes'])

def save_community_db(df, data_storage_root_dir):
    """Saves the detected communities DataFrame to a Parquet file."""
    os.makedirs(data_storage_root_dir, exist_ok=True)
    db_path = os.path.join(data_storage_root_dir, 'communities.parquet')
    df.to_parquet(db_path, index=False)

def load_community_summaries_db(data_storage_root_dir):
    """Loads the community summaries DataFrame from a Parquet file."""
    db_path = os.path.join(data_storage_root_dir, 'community_summaries.parquet')
    if os.path.exists(db_path):
        return pd.read_parquet(db_path)
    return pd.DataFrame(columns=['community_id', 'summary', 'key_entities'])

def save_community_summaries_db(df, data_storage_root_dir):
    """Saves the community summaries DataFrame to a Parquet file."""
    os.makedirs(data_storage_root_dir, exist_ok=True)
    db_path = os.path.join(data_storage_root_dir, 'community_summaries.parquet')
    df.to_parquet(db_path, index=False)