import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

class DataPreprocessor:
    """Class for data preprocessing and preparation"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.scaler = None
        self.feature_names = None
    
    def load_datasets(self, file1, file2):
        """Load and combine two datasets"""
        print("=" * 80)
        print("LOADING DATASETS")
        print("=" * 80)
        
        try:
            df1 = pd.read_csv(os.path.join(self.data_dir, file1))
            df2 = pd.read_csv(os.path.join(self.data_dir, file2))
            
            print(f"\nDataset 1 ({file1}) shape: {df1.shape}")
            print(f"Dataset 2 ({file2}) shape: {df2.shape}")
            
            # Combine datasets
            combined_df = pd.concat([df1, df2], ignore_index=True)
            print(f"\nCombined dataset shape (before): {combined_df.shape}")
            
            # Remove duplicates
            combined_df = combined_df.drop_duplicates()
            print(f"Combined dataset shape (after removing duplicates): {combined_df.shape}")
            
            return combined_df
        
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print(f"Please ensure CSV files are in '{self.data_dir}/' folder")
            return None
    
    def clean_data(self, df):
        """Clean and preprocess data"""
        print("\n" + "=" * 80)
        print("DATA CLEANING")
        print("=" * 80)
        
        # Remove ID column
        df = df.drop('id', axis=1, errors='ignore')
        print("✓ Removed ID column")
        
        # Remove unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        print("✓ Removed unnamed columns")
        
        # Remove duplicate columns
        df = df.iloc[:, ~df.columns.duplicated()]
        print("✓ Removed duplicate columns")
        
        # Remove rows with missing values
        initial_rows = len(df)
        df = df.dropna()
        removed_rows = initial_rows - len(df)
        print(f"✓ Removed {removed_rows} rows with missing values")
        
        print(f"\nCleaned dataset shape: {df.shape}")
        print(f"Number of features: {df.shape[1] - 1}")
        
        return df
    
    def separate_features_target(self, df):
        """Separate features and target variable"""
        print("\n" + "=" * 80)
        print("SEPARATING FEATURES AND TARGET")
        print("=" * 80)
        
        X = df.drop('diagnosis', axis=1)
        y = df['diagnosis']
        
        # Encode target: M=1 (Malignant), B=0 (Benign)
        y = y.map({'M': 1, 'B': 0})
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        print("\nClass distribution:")
        print(f"  Benign (0): {sum(y == 0)} ({sum(y == 0)/len(y)*100:.1f}%)")
        print(f"  Malignant (1): {sum(y == 1)} ({sum(y == 1)/len(y)*100:.1f}%)")
        print(f"  Class ratio: {sum(y == 0) / sum(y == 1):.2f}:1")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def scale_features(self, X_train, X_test=None):
        """Scale features using StandardScaler"""
        print("\n" + "=" * 80)
        print("FEATURE SCALING (Z-SCORE NORMALIZATION)")
        print("=" * 80)
        
        # Create scaler
        self.scaler = StandardScaler()
        
        # Fit on training data and transform
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        print(f"Scaler fitted on training data")
        print(f"Mean of scaled features: {X_train_scaled.mean(axis=0).mean():.6f}")
        print(f"Std of scaled features: {X_train_scaled.std(axis=0).mean():.6f}")
        
        # Transform test data if provided
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        print("\n" + "=" * 80)
        print("TRAIN-TEST SPLIT")
        print("=" * 80)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nTrain set size: {X_train.shape[0]} samples ({(1-test_size)*100:.0f}%)")
        print(f"Test set size: {X_test.shape[0]} samples ({test_size*100:.0f}%)")
        print(f"Number of features: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_pipeline(self, file1, file2, test_size=0.2):
        """Complete preprocessing pipeline"""
        print("\nSTARTING PREPROCESSING PIPELINE\n")
        
        # Load data
        combined_df = self.load_datasets(file1, file2)
        if combined_df is None:
            return None, None, None, None, None, None
        
        # Clean data
        df = self.clean_data(combined_df)
        
        # Separate features and target
        X, y = self.separate_features_target(df)
        
        # Split data (before scaling to avoid data leakage)
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size=test_size)
        
        # Scale features (fit on training, transform test)
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        print("\n" + "=" * 80)
        print("PREPROCESSING COMPLETE")
        print("=" * 80)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, self.scaler, self.feature_names

