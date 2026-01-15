import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    layout="wide", 
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #60A5FA;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #9CA3AF;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #F3F4F6;
        border-bottom: 2px solid #3B82F6;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    /* Fix metric cards visibility */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.8rem !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(path='customer_churn_large_dataset.xlsx'):
    """Load dataset from file path."""
    if os.path.exists(path):
        return pd.read_excel(path)
    return None

@st.cache_data
def load_precomputed_metrics():
    """Load pre-computed metrics if available."""
    if os.path.exists('model_metrics.json'):
        with open('model_metrics.json', 'r') as f:
            return json.load(f)
    return None

@st.cache_resource
def load_models():
    """Load saved ML models."""
    models = {}
    if os.path.exists('customer_churn_classifier.pkl'):
        try:
            models['xgb_sklearn'] = joblib.load('customer_churn_classifier.pkl')
        except Exception:
            models['xgb_sklearn'] = None
    if os.path.exists('ChurnClassifier.h5'):
        try:
            from tensorflow.keras.models import load_model
            models['keras_nn'] = load_model('ChurnClassifier.h5')
        except Exception:
            models['keras_nn'] = None
    return models

def preprocess(df):
    """Preprocess data for model evaluation."""
    df = df.copy()
    # Drop identifier columns if present
    for c in ['CustomerID', 'Name']:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)
    # Ensure expected columns exist
    if 'Churn' not in df.columns:
        st.error('Dataset must contain `Churn` column for evaluation')
        return None, None, None, None
    # One-hot encode categorical cols
    df = pd.get_dummies(df, columns=[c for c in ['Gender', 'Location'] if c in df.columns], drop_first=True)
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Scale numeric columns if present
    num_cols = [c for c in ['Age', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB'] if c in X.columns]
    scaler = MinMaxScaler()
    if num_cols:
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])
    return X_train, X_test, y_train, y_test

def compute_metrics(model, X_test, y_test, model_type='sklearn', threshold=0.5):
    """Compute evaluation metrics for a model."""
    if model is None:
        return None
    if model_type == 'sklearn':
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_test)[:, 1]
        else:
            proba = model.predict(X_test)
    else:
        # keras model: align features to expected input dim
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
    preds = (proba >= threshold).astype(int)
    metrics = {
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds, zero_division=0),
        'recall': recall_score(y_test, preds, zero_division=0),
        'f1': f1_score(y_test, preds, zero_division=0),
        'proba': proba,
        'preds': preds
    }
    return metrics

def plot_confusion(y_true, y_pred):
    """Create confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Not Churned', 'Churned'],
                yticklabels=['Not Churned', 'Churned'])
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_roc(y_true, proba):
    """Create ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color='#3B82F6', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7)
    ax.fill_between(fpr, tpr, alpha=0.3, color='#3B82F6')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def main():
    # Header Section
    st.markdown('<h1 class="main-header">📊 Customer Churn Prediction Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Machine Learning Model Evaluation & Analysis</p>', unsafe_allow_html=True)
    
    # Sidebar Configuration
    st.sidebar.image("https://img.icons8.com/fluency/96/graph-report.png", width=80)
    st.sidebar.title('⚙️ Dashboard Settings')
    st.sidebar.markdown("---")
    
    # Load data and models
    df = load_data()
    models = load_models()
    precomputed_metrics = load_precomputed_metrics()

    # Data Loading Section in Sidebar
    st.sidebar.subheader("📁 Data Source")
    if df is None:
        st.sidebar.warning('No local dataset found')
        uploaded = st.sidebar.file_uploader('Upload Dataset', type=['xlsx', 'csv'])
        if uploaded is None:
            st.info('👆 Please upload a dataset (Excel or CSV) using the sidebar to get started.')
            st.stop()
        try:
            if uploaded.name.lower().endswith('.csv'):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.sidebar.success(f'✅ Loaded: {uploaded.name}')
        except Exception as e:
            st.error(f'Could not read uploaded file: {e}')
            st.stop()
    else:
        st.sidebar.success('✅ Dataset loaded successfully')
        st.sidebar.info(f'📊 Rows: {len(df):,} | Columns: {len(df.columns)}')

    # Model Selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Model Selection")
    available_models = [k for k, v in models.items() if v is not None]
    
    model_names = {
        'xgb_sklearn': '🌲 XGBoost Classifier',
        'keras_nn': '🧠 Neural Network (Keras)'
    }
    
    if not available_models:
        st.sidebar.warning('No saved models found')
        model_choice = None
    else:
        model_choice = st.sidebar.selectbox(
            'Select Model',
            options=available_models,
            format_func=lambda x: model_names.get(x, x)
        )
    
    # Threshold slider
    threshold = st.sidebar.slider('📏 Classification Threshold', 0.0, 1.0, 0.5, 0.01)
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess(df)
    if X_test is None:
        st.stop()

    # Main Content Area
    # Tab Layout for better organization
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🎯 Model Performance", "📊 Visualizations", "📋 Data Explorer"])
    
    # TAB 1: Overview
    with tab1:
        st.markdown('<h3 class="section-header">Dataset Overview</h3>', unsafe_allow_html=True)
        
        # Quick Stats
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Features", f"{len(df.columns) - 1}")
        with col3:
            churn_rate = df['Churn'].mean() * 100 if 'Churn' in df.columns else 0
            st.metric("Churn Rate", f"{churn_rate:.1f}%")
        with col4:
            st.metric("Train Size", f"{len(X_train):,}")
        with col5:
            st.metric("Test Size", f"{len(X_test):,}")
        
        st.markdown("---")
        
        # Churn Distribution
        col_left, col_right = st.columns([1, 2])
        
        with col_left:
            st.markdown("#### 🎯 Churn Distribution")
            if 'Churn' in df.columns:
                churn_counts = df['Churn'].value_counts()
                fig, ax = plt.subplots(figsize=(4, 4))
                colors = ['#10B981', '#EF4444']
                ax.pie(churn_counts, labels=['Retained', 'Churned'], autopct='%1.1f%%', 
                       colors=colors, explode=(0, 0.05), shadow=True, startangle=90)
                ax.set_title('Customer Churn Distribution')
                st.pyplot(fig)
                plt.close()
        
        with col_right:
            st.markdown("#### 📊 Feature Statistics")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'CustomerID' in numeric_cols:
                numeric_cols.remove('CustomerID')
            if 'Churn' in numeric_cols:
                numeric_cols.remove('Churn')
            if numeric_cols:
                stats_df = df[numeric_cols].describe().T
                stats_df = stats_df[['mean', 'std', 'min', '50%', 'max']]
                stats_df.columns = ['Mean', 'Std Dev', 'Min', 'Median', 'Max']
                st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)
    
    # TAB 2: Model Performance
    with tab2:
        st.markdown('<h3 class="section-header">Model Performance Metrics</h3>', unsafe_allow_html=True)
        
        if model_choice and model_choice in models and models.get(model_choice) is not None:
            model = models[model_choice]
            model_type = 'sklearn' if model_choice == 'xgb_sklearn' else 'keras'
            
            # Align features for sklearn model
            X_test_aligned = X_test.copy()
            try:
                feature_names = getattr(model, 'feature_names_in_', None)
                if feature_names is not None:
                    for c in feature_names:
                        if c not in X_test_aligned.columns:
                            X_test_aligned[c] = 0
                    X_test_aligned = X_test_aligned[feature_names]
            except Exception:
                pass

            metrics = compute_metrics(model, X_test_aligned, y_test, 
                                       model_type=model_type, 
                                       threshold=threshold)
            
            if metrics is None:
                st.error('Could not compute metrics for selected model')
                st.stop()

            # Display Model Info
            st.info(f"**Selected Model:** {model_names.get(model_choice, model_choice)} | **Threshold:** {threshold}")
            
            # Metrics Cards
            st.markdown("#### 📊 Evaluation Metrics")
            m1, m2, m3, m4 = st.columns(4)
            
            with m1:
                st.metric("🎯 Accuracy", f"{metrics['accuracy']*100:.2f}%",
                         delta=f"{(metrics['accuracy']-0.5)*100:+.1f}% vs random")
            with m2:
                st.metric("🔍 Precision", f"{metrics['precision']*100:.2f}%")
            with m3:
                st.metric("📣 Recall", f"{metrics['recall']*100:.2f}%")
            with m4:
                st.metric("⚖️ F1 Score", f"{metrics['f1']*100:.2f}%")
            
            st.markdown("---")
            
            # Confusion Matrix and ROC side by side
            st.markdown("#### 📉 Performance Visualizations")
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                st.pyplot(plot_confusion(y_test, metrics['preds']))
            
            with viz_col2:
                st.pyplot(plot_roc(y_test, metrics['proba']))
            
            # Feature Importance (if available)
            if hasattr(model, 'feature_importances_'):
                st.markdown("---")
                st.markdown("#### 🏆 Feature Importance")
                try:
                    fi = model.feature_importances_
                    if hasattr(model, 'feature_names_in_'):
                        names = list(model.feature_names_in_)
                    else:
                        names = list(X_test_aligned.columns)
                    
                    fi_df = pd.DataFrame({'Feature': names, 'Importance': fi})
                    fi_df = fi_df.sort_values('Importance', ascending=True)
                    
                    fig, ax = plt.subplots(figsize=(10, max(4, len(fi_df) * 0.4)))
                    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(fi_df)))
                    bars = ax.barh(fi_df['Feature'], fi_df['Importance'], color=colors)
                    ax.set_xlabel('Importance Score', fontsize=12)
                    ax.set_title('Feature Importance Ranking', fontsize=14, fontweight='bold')
                    ax.grid(axis='x', alpha=0.3)
                    
                    # Add value labels on bars
                    for bar, val in zip(bars, fi_df['Importance']):
                        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                               f'{val:.3f}', va='center', fontsize=10)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.info(f'Feature importance visualization not available: {e}')
        else:
            st.warning('👈 Select a model from the sidebar to view performance metrics.')
            
            # Show precomputed metrics if available
            if precomputed_metrics:
                st.markdown("#### 📋 Pre-computed Metrics (from model_metrics.json)")
                for model_name, mets in precomputed_metrics.items():
                    with st.expander(f"📊 {model_name}"):
                        cols = st.columns(4)
                        cols[0].metric("Accuracy", f"{mets.get('accuracy', 0)*100:.2f}%")
                        cols[1].metric("Precision", f"{mets.get('precision', 0)*100:.2f}%")
                        cols[2].metric("Recall", f"{mets.get('recall', 0)*100:.2f}%")
                        cols[3].metric("F1", f"{mets.get('f1', 0)*100:.2f}%")
    
    # TAB 3: Visualizations
    with tab3:
        st.markdown('<h3 class="section-header">Data Visualizations</h3>', unsafe_allow_html=True)
        
        viz_option = st.selectbox("Select Visualization", [
            "Age Distribution",
            "Monthly Bill Distribution", 
            "Usage by Churn Status",
            "Correlation Heatmap"
        ])
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        if viz_option == "Age Distribution":
            if 'Age' in df.columns and 'Churn' in df.columns:
                sns.histplot(data=df, x='Age', hue='Churn', kde=True, ax=ax, 
                           palette=['#10B981', '#EF4444'])
                ax.set_title('Age Distribution by Churn Status', fontsize=14)
                ax.legend(['Retained', 'Churned'])
            else:
                ax.text(0.5, 0.5, 'Required columns not found', ha='center', va='center')
        
        elif viz_option == "Monthly Bill Distribution":
            if 'Monthly_Bill' in df.columns and 'Churn' in df.columns:
                sns.boxplot(data=df, x='Churn', y='Monthly_Bill', ax=ax,
                          palette=['#10B981', '#EF4444'])
                ax.set_xticklabels(['Retained', 'Churned'])
                ax.set_title('Monthly Bill by Churn Status', fontsize=14)
            else:
                ax.text(0.5, 0.5, 'Required columns not found', ha='center', va='center')
        
        elif viz_option == "Usage by Churn Status":
            if 'Total_Usage_GB' in df.columns and 'Churn' in df.columns:
                sns.violinplot(data=df, x='Churn', y='Total_Usage_GB', ax=ax,
                             palette=['#10B981', '#EF4444'])
                ax.set_xticklabels(['Retained', 'Churned'])
                ax.set_title('Total Usage (GB) by Churn Status', fontsize=14)
            else:
                ax.text(0.5, 0.5, 'Required columns not found', ha='center', va='center')
        
        elif viz_option == "Correlation Heatmap":
            numeric_df = df.select_dtypes(include=[np.number])
            cols_to_drop = ['CustomerID']
            numeric_df = numeric_df.drop(columns=[c for c in cols_to_drop if c in numeric_df.columns])
            corr = numeric_df.corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax, fmt='.2f')
            ax.set_title('Feature Correlation Matrix', fontsize=14)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # TAB 4: Data Explorer
    with tab4:
        st.markdown('<h3 class="section-header">Data Explorer</h3>', unsafe_allow_html=True)
        
        col_left, col_right = st.columns([1, 3])
        
        with col_left:
            st.markdown("#### Filter Options")
            show_rows = st.slider("Rows to display", 5, 100, 20)
            
            if 'Churn' in df.columns:
                churn_filter = st.radio("Filter by Churn", ['All', 'Churned Only', 'Retained Only'])
            else:
                churn_filter = 'All'
        
        with col_right:
            filtered_df = df.copy()
            if churn_filter == 'Churned Only' and 'Churn' in df.columns:
                filtered_df = df[df['Churn'] == 1]
            elif churn_filter == 'Retained Only' and 'Churn' in df.columns:
                filtered_df = df[df['Churn'] == 0]
            
            st.markdown(f"**Showing {min(show_rows, len(filtered_df))} of {len(filtered_df):,} records**")
            st.dataframe(filtered_df.head(show_rows), use_container_width=True)
            
            # Download button
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Filtered Data as CSV",
                csv,
                "churn_data_filtered.csv",
                "text/csv",
                key='download-csv'
            )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #6B7280; padding: 1rem;'>
            <p>🚀 Customer Churn Prediction Dashboard | Built with Streamlit</p>
            <p style='font-size: 0.8rem;'>Models: XGBoost Classifier & Keras Neural Network</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()
