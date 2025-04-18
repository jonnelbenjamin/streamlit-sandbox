# components/h2o_automl.py

import streamlit as st
import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML
import matplotlib.pyplot as plt
import time
import os
import tempfile

def train_automl_models(df, h2o_df, target_col, selected_features, max_models, max_runtime_secs, problem_type, seed):
    """Handles the AutoML model training process"""
    # Set up the target and features
    y = target_col
    x = selected_features
    
    is_numeric_target = df[target_col].dtype in [np.int64, np.float64]
    
    # Log transformation for target variable option
    if is_numeric_target and problem_type != "Classification":
        log_transform = st.checkbox("Apply log transformation to target (useful for skewed data)")
        if log_transform and df[target_col].min() > 0:
            h2o_df[y] = h2o_df[y].log()
            st.info("Log transformation applied to target variable.")
    
    # Show which algorithms will be used
    st.subheader("Algorithms that will be trained")
    algo_cols = st.columns(3)
    with algo_cols[0]:
        st.markdown("✅ **Gradient Boosting Machines**")
        st.markdown("✅ **Random Forest**")
    with algo_cols[1]:
        st.markdown("✅ **Deep Learning**")
        st.markdown("✅ **Generalized Linear Models**")
    with algo_cols[2]:
        st.markdown("✅ **Stacked Ensembles**")
        st.markdown("✅ **XGBoost** (if available)")
    
    # Set up AutoML with more explicit parameters
    aml = H2OAutoML(
        max_models=max_models,
        seed=seed,
        max_runtime_secs=max_runtime_secs,
        sort_metric="AUTO",
        balance_classes=True,
        exclude_algos=None  # Include all algorithms
    )
    
    # Specify problem type if explicitly chosen
    if problem_type == "Regression":
        h2o_df[y] = h2o_df[y].asfactor() if h2o_df[y].is_numeric() else h2o_df[y]
    elif problem_type == "Classification":
        h2o_df[y] = h2o_df[y].asfactor()
    
    # Train the models
    with st.spinner(f"Training up to {max_models} models with a maximum runtime of {max_runtime_secs} seconds..."):
        start_time = time.time()
        aml.train(x=x, y=y, training_frame=h2o_df)
        elapsed_time = time.time() - start_time
    
    st.success(f"AutoML training completed in {elapsed_time:.2f} seconds!")
    
    # Show leaderboard
    st.subheader("Model Leaderboard")
    leaderboard = aml.leaderboard
    st.dataframe(leaderboard.as_data_frame())
    
    # Get the best model
    best_model = aml.leader
    
    # Display model details
    st.subheader("Best Model Details")
    st.write(f"Best model: {best_model.model_id}")
    
    # Show model performance metrics
    st.subheader("Performance Metrics")
    
    # Different metrics for classification and regression
    if best_model.model_category == "Binomial" or best_model.model_category == "Multinomial":
        # Classification metrics
        perf = best_model.model_performance()
        metrics_df = pd.DataFrame({
            'Metric': ['AUC', 'Logloss', 'AUCPR', 'Mean Per-Class Error'],
            'Value': [
                perf.auc(), 
                perf.logloss(),
                perf.aucpr() if hasattr(perf, 'aucpr') else 'N/A',
                perf.mean_per_class_error()
            ]
        })
        st.dataframe(metrics_df)
        
        # Plot confusion matrix
        st.subheader("Confusion Matrix")
        conf_matrix = perf.confusion_matrix().as_data_frame()
        st.dataframe(conf_matrix)
        
    else:
        # Regression metrics
        perf = best_model.model_performance()
        metrics_df = pd.DataFrame({
            'Metric': ['RMSE', 'MSE', 'MAE', 'R²'],
            'Value': [
                perf.rmse(), 
                perf.mse(),
                perf.mae(),
                perf.r2()
            ]
        })
        st.dataframe(metrics_df)
        
        # Scatter plot of actual vs predicted
        st.subheader("Actual vs Predicted")
        preds = best_model.predict(h2o_df)
        pred_df = h2o.as_list(preds)
        actual = h2o.as_list(h2o_df[y])
        
        fig, ax = plt.subplots()
        ax.scatter(actual, pred_df['predict'])
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
        st.pyplot(fig)
    
    # Feature importance
    st.subheader("Feature Importance")
    if hasattr(best_model, 'varimp'):
        varimp = best_model.varimp()
        if varimp is not None:
            # Convert to DataFrame
            varimp_df = pd.DataFrame(varimp, columns=['Feature', 'Importance', 'Percentage'])
            
            # Plot feature importance
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(varimp_df['Feature'][:10], varimp_df['Percentage'][:10])
            ax.set_xlabel('Importance (%)')
            ax.set_title('Top 10 Feature Importance')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Feature importance not available for this model.")
    else:
        st.info("Feature importance not available for this model.")
    
    # Option to download the model
    if st.button("Download Best Model"):
        # Save model to a temporary file
        model_path = h2o.save_model(model=best_model, path="./", force=True)
        st.success(f"Model saved locally at: {model_path}")
        st.info("You can use this model for predictions in production or other applications.")
    
    return best_model

def show():
    st.title("H2O AutoML Playground")
    st.markdown("""
    This component leverages H2O's AutoML to automatically train and tune machine learning models.
    Upload your dataset, select parameters, and let AutoML find the best model for your data.
    """)
    
    # Initialize H2O if not already running
    if not h2o.connection():
        with st.spinner("Starting H2O cluster..."):
            h2o.init()
    
    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Load the data
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.dataframe(df.head())
            
            # Convert to H2O frame - using temp file instead of StringIO
            with st.spinner("Converting data to H2O frame..."):
                # Save DataFrame to a temporary file
                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_file:
                        temp_path = tmp_file.name
                        df.to_csv(temp_path, index=False)
                    
                    # Import the temporary file to H2O
                    h2o_df = h2o.import_file(temp_path)
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.unlink(temp_path)
            
            # Display basic information
            st.write(f"Dataset dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Column selection
            st.subheader("Column Selection")
            
            # Target column selection
            target_col = st.selectbox("Select Target Column", df.columns)
            
            # Feature selection
            st.write("Select Features (or leave all checked to use all features except target)")
            feature_cols = [col for col in df.columns if col != target_col]
            selected_features = st.multiselect("Features", feature_cols, default=feature_cols)
            
            if not selected_features:
                st.warning("Please select at least one feature column.")
                return
            
            # Model training parameters
            st.subheader("Model Training Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                max_models = st.slider("Maximum Models to Train", 1, 20, 5)
                max_runtime_secs = st.slider("Maximum Runtime (seconds)", 10, 3600, 300)
            
            with col2:
                # Determine if regression or classification
                is_numeric_target = df[target_col].dtype in [np.int64, np.float64]
                
                if is_numeric_target:
                    problem_type = st.radio("Problem Type", ["Auto-detect", "Regression", "Classification"])
                else:
                    problem_type = "Classification"
                    st.write("Problem Type: Classification (based on target column)")
                
                seed = st.number_input("Random Seed", value=42)
            
            # Training button
            if st.button("Train Models with AutoML"):
                train_automl_models(df, h2o_df, target_col, selected_features, max_models, 
                                   max_runtime_secs, problem_type, seed)
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Traceback:", exc_info=True)
    
    else:
        # Sample datasets option
        st.subheader("Or try with sample datasets")
        sample_data = st.selectbox(
            "Select a sample dataset",
            ["None", "Iris", "Diabetes", "California Housing", "Wine Quality"]
        )
        
        if sample_data != "None":
            # Load the selected sample dataset
            if sample_data == "Iris":
                from sklearn.datasets import load_iris
                data = load_iris()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df['target'] = data.target
                st.write("Iris dataset loaded (Classification)")
                default_target = 'target'
                problem_type = "Classification"
            elif sample_data == "Diabetes":
                from sklearn.datasets import load_diabetes
                data = load_diabetes()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df['target'] = data.target
                st.write("Diabetes dataset loaded (Regression)")
                default_target = 'target'
                problem_type = "Regression"
            elif sample_data == "California Housing":
                from sklearn.datasets import fetch_california_housing
                data = fetch_california_housing()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df['target'] = data.target
                st.write("California Housing dataset loaded (Regression)")
                default_target = 'target'
                problem_type = "Regression"
            elif sample_data == "Wine Quality":
                from sklearn.datasets import load_wine
                data = load_wine()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df['target'] = data.target
                st.write("Wine Quality dataset loaded (Classification)")
                default_target = 'target'
                problem_type = "Classification"
            
            st.dataframe(df.head())
            
            # Convert to H2O frame using temp file
            with st.spinner("Converting data to H2O frame..."):
                # Save DataFrame to a temporary file
                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_file:
                        temp_path = tmp_file.name
                        df.to_csv(temp_path, index=False)
                    
                    # Import the temporary file to H2O
                    h2o_df = h2o.import_file(temp_path)
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.unlink(temp_path)
            
            # Display basic information
            st.write(f"Dataset dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Column selection with defaults for sample data
            st.subheader("Column Selection")
            
            # Target column selection
            target_col = st.selectbox("Select Target Column", df.columns, index=df.columns.get_loc(default_target))
            
            # Feature selection
            st.write("Select Features (or leave all checked to use all features except target)")
            feature_cols = [col for col in df.columns if col != target_col]
            selected_features = st.multiselect("Features", feature_cols, default=feature_cols)
            
            if not selected_features:
                st.warning("Please select at least one feature column.")
                return
            
            # Model training parameters
            st.subheader("Model Training Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                max_models = st.slider("Maximum Models to Train", 1, 20, 5)
                max_runtime_secs = st.slider("Maximum Runtime (seconds)", 10, 3600, 300)
            
            with col2:
                seed = st.number_input("Random Seed", value=42)
            
            # Training button
            if st.button("Train Models with AutoML"):
                train_automl_models(df, h2o_df, target_col, selected_features, max_models, 
                                   max_runtime_secs, problem_type, seed)

    # Show information about H2O AutoML with more detailed explanations
    with st.expander("About H2O AutoML"):
        st.markdown("""
        ### What is H2O AutoML?
        
        H2O AutoML provides automated machine learning capabilities to find the best performing model for your data.
        It automatically runs through various algorithms, hyperparameters, and feature engineering techniques
        to optimize model performance.
        
        ### Algorithms included in H2O AutoML:
        
        * **Gradient Boosting Machines (GBM)**: Builds an ensemble of decision trees sequentially to correct errors
        * **Random Forest**: Creates multiple decision trees and merges their predictions
        * **Deep Learning (Neural Networks)**: Multi-layer perceptron neural networks with various architectures
        * **Generalized Linear Models (GLM)**: Extensions of linear regression for different distribution types
        * **Stacked Ensembles**: Combines predictions from multiple models for improved accuracy
        * **XGBoost**: High-performance implementation of gradient boosting (if available in your environment)
        
        ### Key Features:
        
        * **Automatic feature engineering**: Transforms variables to improve model performance
        * **Cross-validation**: Uses k-fold validation for robust model evaluation
        * **Leaderboard ranking**: Compares models using metrics appropriate for your problem
        * **Ensemble methods**: Creates stacked ensembles that combine multiple base models
        * **Model explanations**: Provides variable importance and other model insights
        * **Hyperparameter tuning**: Automatically searches for optimal parameters
        """)
        
    # Add a shutdown button for the H2O cluster
    st.sidebar.markdown("---")
    if st.sidebar.button("Shutdown H2O Cluster"):
        h2o.cluster().shutdown()
        st.sidebar.success("H2O cluster shut down successfully.")