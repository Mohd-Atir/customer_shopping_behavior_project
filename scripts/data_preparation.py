import pandas as pd
from sqlalchemy import create_engine
import os
import logging
import numpy as np


try:
    if os.makedirs('../logs', exist_ok=True):
        logger.info("Directory created or already exist")
except Exceptin as e:
    logger.error("Error creating directory", e)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("../logs/data_preparation.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# -----------------------------------
# 1. Load Data
# -----------------------------------

def load_raw_data(file_path):
    '''This function will load the raw data'''
    logger.info(f"{'-'*25} Loading the data {'-'*25}")

    try:   
        df = pd.read_csv(file_path)
        logger.info(f"{'-'*25} Initial data preview {'-'*25} \n{df.head(5)}")
        return df

    except Exception:
        logger.error("Error while loading raw data", exc_info=True)
        raise

# -----------------------------------
# 2. Standardize Column Names
# -----------------------------------

def clean_column_name(df, rename_column: dict = None):
    """
    Standardize DataFrame column names and optionally rename specific columns.

    Steps:
    - Lowercase all column names
    - Strip leading/trailing spaces
    - Replace spaces with underscores
    - Apply custom renames if provided

    Args:
        df : Input DataFrame
        rename_map (dict, optional): Mapping of old_name -> new_name after cleaning
    
    Returns:
        df: DataFrame with cleaned column names
    """
    
    logger.info(f"{'-'*25} Cleaning columns name {'-'*25}")

    try:
        before = df.columns
    
        # General cleaning
        df.columns = (
            df.columns
            .str.lower()
            .str.strip()
            .str.replace(' ', '_')
        )
    
        # Renames column name if provided
        if rename_column:
            df.rename(columns=rename_column, inplace=True)
    
        after = df.columns
        logger.info(f"{'-'*25} Columns name before cleaning: {'-'*25} \n{before} \n{'-'*25} and after cleaning: {'-'*25} \n{after}")
        return df

    except Exception:
        logger.error("Error while cleaning columns", exc_info=True)
        raise

# -----------------------------------
# 3. Handle Missing Values
# -----------------------------------

def handle_missing_values(df):
    """
        This function handles missing values of the columns
        Missing Value Strategy:
        - review_rating -> category-wise median
        - numeric columns -> global median
        - categorical columns -> mode
        This approach preserves business context and avoids bias.
    """

    logger.info(f"{'-'*25} Handling Missing Values {'-'*25}")

    try:
        # 1- Review Rating -> Category-Wise Median
        if 'review_rating' in df.columns and 'category' in df.columns:
            before = df['review_rating'].isnull().sum()
            
            df['review_rating'] = (
                df.groupby('category')['review_rating']
                .transform(lambda x: x.fillna(x.median()))
            )
    
            after = df['review_rating'].isnull().sum()
            logger.info(f"{'-'*25}review_rating nulls:{'-'*25} {before} -> {after}")
    
        # 2- Other numeric Columns -> Global Median
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        numeric_cols = numeric_cols.drop('review_rating', errors='ignore')
    
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
                logger.info(f"Filled nulls in numeric column: {col}")
    
    
        # 3- Categorical Columns -> Mode
        categorical_cols = df.select_dtypes(include=['object']).columns
    
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])
                logger.info(f"Filled nulls in categorical column: {col}")
    
        return df

    except Exception:
        logger.error("Error while handling missing values", exc_info=True)
        raise

# -----------------------------------
# 4. Feature Engineering
# -----------------------------------

def create_age_group(df):
    '''This function creates age_group column'''
    logger.info(f"{'-'*25} Creating age_group column {'-'*25}")

    try:
        before = df.columns
        
        bins = [0, 25, 40, 60, 100]
        labels = ['Young Adult', 'Adult', 'Middle Aged', 'Senior']
        df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
        
        after = df.columns
        logger.info(f"{'-'*25} Columns name before creating age_group: {'-'*25} \n{before} \n{'-'*25} and after creating age_group: {'-'*25} \n{after}")
        
        return df
        
    except Exception:
        logger.error("Customer age_group column creation failed", exc_info=True)
        raise    

# -----------------------------------
# 5. Customer segment creation
# -----------------------------------

def create_customer_segment(df):
    """
    Create customer segmentation based on purchase frequency.

    Segmentation Logic:
    - New: previous_purchases == 1
    - Returning: previous_purchases between 2 and 10
    - Loyal: previous_purchases > 10
    """
    logger.info(f"{'-'*25} Creating customer_segment {'-'*25}")
    
    try:
        if 'previous_purchases' not in df.columns:
            raise KeyError("Column 'previous_purchases' not found in DataFrame")

        conditions = [
            df['previous_purchases'] == 1,
            df['previous_purchases'].between(2, 10),
            df['previous_purchases'] > 10
        ]

        choices = ['New', 'Returning', 'Loyal']

        df['customer_segment'] = np.select(
            conditions,
            choices,
            default='Unknown'
        )

        return df

    except Exception:
        logger.error("Customer segment creation failed", exc_info=True)
        raise    


# -----------------------------------
# 6. Final Data Preparation
# -----------------------------------

def prepare_data(file_path):
    '''This function prepares the data'''
    logger.info(f"{'-'*25} Preparing data {'-'*25}")

    try:     
        df = load_raw_data(file_path)
        df = clean_column_name(df, rename_column={'purchase_amount_(usd)': 'purchase_amount'})
        df = handle_missing_values(df)
        df = create_age_group(df)
        df = create_customer_segment(df)
        return df
        
    except Exception:
        logger.error("Data preparation failed", exc_info=True)
        raise    

# -----------------------------------
# 7. Ingest Data into Database
# -----------------------------------

def ingest_db(df, table_name, username, password, host, port, db_name):
    logger.info(f"{'-'*25} Starting database ingestion {'-'*25}")
    engine = create_engine(
        f"mysql+pymysql://{username}:{password}@{host}:{port}/{db_name}"
    )

    try:
        df.to_sql(
            name=table_name,
            con=engine,
            if_exists='replace',
            index=False
        )
        logger.info(f"Data successfully inserted into table: {table_name}")

    except Exception as e:
        logger.error("Database ingestion failed", exc_info=True)
        raise

# -----------------------------------
# 8. Pipeline Execution
# -----------------------------------

if __name__ == "__main__":
    FILE_PATH = "../data/customer_shopping_behavior.csv"

    DB_CONFIG = {
        "table_name": "customer_shopping_behavior",
        "username": "root",
        "password": "",
        "host": "localhost",
        "port": 3306,
        "db_name": "customer"
    }

    logger.info(f"{'-'*25} Pipeline started {'-'*25}")
    
    final_df = prepare_data(FILE_PATH)
    ingest_db(final_df, **DB_CONFIG)

    logger.info(f"{'-'*25} Pipeline completed successfully {'-'*25}")

    





