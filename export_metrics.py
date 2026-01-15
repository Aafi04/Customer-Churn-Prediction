"""Export model evaluation metrics and feature importances to files.

Usage:
    python export_metrics.py

Outputs:
    - model_metrics.json
    - feature_importances_<model>.csv (if available)
"""
import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_data(path='customer_churn_large_dataset.xlsx'):
    if os.path.exists(path):
        if path.lower().endswith('.csv'):
            return pd.read_csv(path)
        return pd.read_excel(path)
    raise FileNotFoundError(f'{path} not found in workspace')


def preprocess(df):
    df = df.copy()
    for c in ['CustomerID', 'Name']:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)
    if 'Churn' not in df.columns:
        raise ValueError('Dataset must contain `Churn` column')
    df = pd.get_dummies(df, columns=[c for c in ['Gender', 'Location'] if c in df.columns], drop_first=True)
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    num_cols = [c for c in ['Age', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB'] if c in X.columns]
    if num_cols:
        scaler = MinMaxScaler()
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])
    return X_train, X_test, y_train, y_test


def compute_basic_metrics(y_true, preds):
    return {
        'accuracy': float(accuracy_score(y_true, preds)),
        'precision': float(precision_score(y_true, preds, zero_division=0)),
        'recall': float(recall_score(y_true, preds, zero_division=0)),
        'f1': float(f1_score(y_true, preds, zero_division=0))
    }


def main():
    metrics = {}
    try:
        df = load_data()
    except Exception as e:
        print('Could not load dataset:', e)
        return

    X_train, X_test, y_train, y_test = preprocess(df)

    # Check for sklearn model
    if os.path.exists('customer_churn_classifier.pkl'):
        try:
            model = joblib.load('customer_churn_classifier.pkl')
            proba = None
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test)[:, 1]
                preds = (proba >= 0.5).astype(int)
            else:
                preds = model.predict(X_test)
            metrics['customer_churn_classifier.pkl'] = compute_basic_metrics(y_test, preds)
            # feature importance
            if hasattr(model, 'feature_importances_'):
                fi = getattr(model, 'feature_importances_')
                names = list(getattr(model, 'feature_names_in_', X_test.columns))
                fi_df = pd.DataFrame({'feature': names, 'importance': fi}).sort_values('importance', ascending=False)
                fi_df.to_csv('feature_importances_customer_churn_classifier.csv', index=False)
        except Exception as e:
            print('Failed to evaluate sklearn model:', e)

    # Check for keras model
    if os.path.exists('ChurnClassifier.h5'):
        try:
            from tensorflow.keras.models import load_model
            model = load_model('ChurnClassifier.h5')
            # detect expected input dim
            input_dim = None
            try:
                input_shape = model.input_shape
                if isinstance(input_shape, (list, tuple)):
                    if isinstance(input_shape[0], tuple):
                        input_dim = input_shape[0][-1]
                    else:
                        input_dim = input_shape[-1]
            except Exception:
                input_dim = None

            X_for_model = X_test.copy()
            if input_dim is not None and X_for_model.shape[1] != input_dim:
                preferred = ['Monthly_Bill', 'Total_Usage_GB', 'Age', 'Subscription_Length_Months']
                selected = [c for c in preferred if c in X_for_model.columns]
                if len(selected) == input_dim:
                    X_for_model = X_for_model[selected]
                else:
                    if X_for_model.shape[1] >= input_dim:
                        X_for_model = X_for_model.iloc[:, :input_dim]
                    else:
                        raise ValueError(f'Keras model expects {input_dim} features but input has {X_for_model.shape[1]}')

            proba = model.predict(X_for_model).ravel()
            preds = (proba >= 0.5).astype(int)
            metrics['ChurnClassifier.h5'] = compute_basic_metrics(y_test, preds)
        except Exception as e:
            print('Failed to evaluate keras model:', e)

    # Save metrics
    if metrics:
        with open('model_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        print('Wrote model_metrics.json and any feature_importances_*.csv files')
    else:
        print('No models evaluated. Place saved models in workspace and re-run.')


if __name__ == '__main__':
    main()
