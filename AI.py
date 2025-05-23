import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE

# Configuration
MODEL_FILENAME = 'autism_rf_model.joblib'
TRAIN_DATA_FILENAME = 'training_data.csv'
TEST_DATA_FILENAME = 'test_data.csv'
LABEL_ENCODERS_FILENAME = 'label_encoders.joblib'
RANDOM_STATE = 42
TEST_SIZE = 0.2


def clean_autism_column(df):
    """Standardize autism column to only 'yes' and 'no'"""
    if 'autism' in df.columns:
        df['autism'] = df['autism'].str.lower().str.strip()

        df['autism'] = df['autism'].replace({
            'y': 'yes',
            'ye': 'yes',
            'yes': 'yes',
            'n': 'no',
            'no': 'no',
            'non': 'no'
        })

        valid_values = ['yes', 'no']
        df = df[df['autism'].isin(valid_values)]

        df['autism'] = df['autism'].map({'no': 0, 'yes': 1})

    return df


def save_model_artifacts(model, X_train, X_test, y_train, y_test, label_encoders):
    """Save all model artifacts to disk"""
    # Save model
    joblib.dump(model, MODEL_FILENAME)

    train_data = X_train.copy()
    train_data['autism'] = y_train
    train_data.to_csv(TRAIN_DATA_FILENAME, index=False)

    test_data = X_test.copy()
    test_data['autism'] = y_test
    test_data.to_csv(TEST_DATA_FILENAME, index=False)

    joblib.dump(label_encoders, LABEL_ENCODERS_FILENAME)

    print(
        f"Model artifacts saved to disk: {MODEL_FILENAME}, {TRAIN_DATA_FILENAME}, {TEST_DATA_FILENAME}, {LABEL_ENCODERS_FILENAME}")


def load_model_artifacts():
    """Load all model artifacts from disk"""
    required_files = [MODEL_FILENAME, TRAIN_DATA_FILENAME, TEST_DATA_FILENAME, LABEL_ENCODERS_FILENAME]

    if not all(os.path.exists(f) for f in required_files):
        raise FileNotFoundError("One or more model artifact files are missing")

    model = joblib.load(MODEL_FILENAME)
    train_data = pd.read_csv(TRAIN_DATA_FILENAME)
    test_data = pd.read_csv(TEST_DATA_FILENAME)
    label_encoders = joblib.load(LABEL_ENCODERS_FILENAME)

    # Check for additional artifacts
    if os.path.exists('model_artifacts.joblib'):
        print("Loading additional model artifacts...")
        artifacts = joblib.load('model_artifacts.joblib')

        # Store artifacts in model's metadata for easy access
        if not hasattr(model, 'metadata'):
            model.metadata = {}

        model.metadata.update({
            'scaler': artifacts.get('scaler'),
            'selector': artifacts.get('selector'),
            'selected_features': artifacts.get('selected_features')
        })

    return model, train_data, test_data, label_encoders


def evaluate_model(model=None, test_data=None):
    """Evaluate the model on test data and print detailed metrics"""
    if model is None or test_data is None:
        print("Loading model and test data...")
        model, train_data, test_data, _ = load_model_artifacts()
    else:
        # We need train_data for column reference
        _, train_data, _, _ = load_model_artifacts()

    # Drop rows with missing values in the target column
    test_data = test_data.dropna(subset=['autism'])
    print(f"Test data shape after dropping rows with missing target: {test_data.shape}")

    # Extract features and target
    X_test = test_data.drop('autism', axis=1)
    y_test = test_data['autism']

    # Check if we have additional artifacts
    artifacts = None
    if os.path.exists('model_artifacts.joblib'):
        artifacts = joblib.load('model_artifacts.joblib')

    # Apply preprocessing if artifacts exist
    if artifacts:
        scaler = artifacts.get('scaler')
        selector = artifacts.get('selector')
        selected_features = artifacts.get('selected_features')

        if scaler and selector and selected_features is not None:
            # Print column information for debugging
            print(f"Test data columns: {X_test.columns.tolist()}")
            print(f"Selected features: {selected_features.tolist()}")

            # Since we're using the model directly with the selected features,
            # we only need to ensure the test data has those selected features

            # Create a new DataFrame with just the selected features
            processed_data = pd.DataFrame()

            # For each selected feature, either copy it from X_test or use a default value
            for col in selected_features:
                if col in X_test.columns:
                    processed_data[col] = X_test[col]
                else:
                    # Use the mean from training data for missing columns
                    processed_data[col] = train_data[col].mean()

            # Fill any remaining missing values with the mean of training data
            processed_data.fillna(train_data.mean(), inplace=True)

            # Verify that all selected features are present
            assert set(processed_data.columns) == set(selected_features), "Selected feature mismatch after processing"

            # Make predictions directly using the processed data
            # No need to apply scaler or selector again as we're already using the selected features
            y_pred = model.predict(processed_data)
            y_pred_proba = model.predict_proba(processed_data)[:, 1]
        else:
            # Legacy prediction
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        # Legacy prediction
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate ROC AUC
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        # Calculate precision-recall AUC
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)

        print(f'ROC AUC: {roc_auc:.4f}')
        print(f'PR AUC: {pr_auc:.4f}')
    except Exception as e:
        print(f"Could not calculate ROC AUC: {e}")

    print(f'Accuracy: {accuracy:.4f}')
    print(classification_report(y_test, y_pred))

    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"Sensitivity (True Positive Rate): {sensitivity:.4f}")
    print(f"Specificity (True Negative Rate): {specificity:.4f}")

    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc if 'roc_auc' in locals() else None,
        'pr_auc': pr_auc if 'pr_auc' in locals() else None,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': cm,
        'classification_report': classification_report(y_test, y_pred)
    }


def train_new_model():
    """Train a new model from scratch and save all artifacts"""
    df = pd.read_csv('eeg_data.csv', low_memory=False)

    df = clean_autism_column(df)

    # Handle missing values
    df = df.dropna(subset=['autism'])  # Drop rows with missing target

    # Encode categorical features
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=['object']).columns.drop('autism', errors='ignore')

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = df[col].fillna('unknown')  # Handle missing values
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Fill remaining missing values with median
    df = df.fillna(df.median())

    print("Columns in dataset:", df.columns.tolist())

    X = df.drop(['autism'], axis=1)
    y = df['autism']

    print("Unique values in target:", y.unique())
    if len(y.unique()) != 2:
        raise ValueError("Target variable must have exactly 2 classes (0 and 1)")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Check for class imbalance
    class_counts = np.bincount(y_train)
    print(f"Class distribution before SMOTE: {class_counts}")

    if class_counts[0] / sum(class_counts) < 0.4 or class_counts[0] / sum(class_counts) > 0.6:
        print("Applying SMOTE to handle class imbalance...")
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        print(f"Class distribution after SMOTE: {np.bincount(y_train)}")

    # Feature selection
    print("Performing feature selection...")
    base_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    base_model.fit(X_train_scaled, y_train)

    # Select features based on importance
    selector = SelectFromModel(base_model, threshold="mean")
    selector.fit(X_train_scaled, y_train)

    # Get selected feature indices and names
    selected_features = X_train_scaled.columns[selector.get_support()]
    print(f"Selected {len(selected_features)} features: {selected_features.tolist()}")

    # Filter data to include only selected features
    X_train_selected = selector.transform(X_train_scaled)
    X_test_selected = selector.transform(X_test_scaled)

    # Convert back to DataFrame with selected feature names
    X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features)
    X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features)

    # Hyperparameter tuning with cross-validation
    print("Performing hyperparameter tuning...")

    # Define models and parameters for grid search
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    gb_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }

    # Create cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Train and tune Random Forest
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        rf_params,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    rf_grid.fit(X_train_selected, y_train)
    print(f"Best Random Forest parameters: {rf_grid.best_params_}")

    # Train and tune Gradient Boosting
    gb_grid = GridSearchCV(
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        gb_params,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    gb_grid.fit(X_train_selected, y_train)
    print(f"Best Gradient Boosting parameters: {gb_grid.best_params_}")

    # Create ensemble model
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf_grid.best_estimator_),
            ('gb', gb_grid.best_estimator_)
        ],
        voting='soft'
    )

    # Train the ensemble model
    ensemble.fit(X_train_selected, y_train)

    # Evaluate on test set
    y_pred = ensemble.predict(X_test_selected)
    y_pred_proba = ensemble.predict_proba(X_test_selected)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Calculate precision-recall AUC
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'ROC AUC: {roc_auc:.4f}')
    print(f'PR AUC: {pr_auc:.4f}')
    print(classification_report(y_test, y_pred))

    # Store additional artifacts for prediction
    artifacts = {
        'scaler': scaler,
        'selector': selector,
        'selected_features': selected_features
    }

    # Save model and artifacts
    joblib.dump(artifacts, 'model_artifacts.joblib')
    save_model_artifacts(ensemble, X_train_selected, X_test_selected, y_train, y_test, label_encoders)

    return ensemble, X_train_selected, X_test_selected, y_train, y_test, label_encoders


def predict_new_data(new_data):
    """Use the saved model to predict new data"""
    model, train_data, test_data, label_encoders = load_model_artifacts()

    # Load additional artifacts
    if os.path.exists('model_artifacts.joblib'):
        artifacts = joblib.load('model_artifacts.joblib')
        scaler = artifacts['scaler']
        selector = artifacts['selector']
        selected_features = artifacts['selected_features']
    else:
        print("Warning: model_artifacts.joblib not found. Using legacy prediction method.")
        scaler = None
        selector = None
        selected_features = None

    # Print column information for debugging
    print(f"Training data columns: {train_data.columns.tolist()}")
    print(f"New data columns: {new_data.columns.tolist()}")

    # Handle categorical features
    for col in new_data.select_dtypes(include=['object']).columns:
        if col in label_encoders:
            le = label_encoders[col]
            # Fill missing values with 'unknown' and handle values not seen during training
            new_data[col] = new_data[col].fillna('unknown')
            new_data[col] = new_data[col].map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    # Create a new DataFrame with all expected columns
    expected_columns = train_data.drop('autism', axis=1).columns
    missing_columns = [col for col in expected_columns if col not in new_data.columns]

    if missing_columns:
        print(f"Missing columns in test data: {missing_columns}")

    # Create a new DataFrame with all the expected columns
    processed_data = pd.DataFrame()

    # Copy existing columns from new_data
    for col in expected_columns:
        if col in new_data.columns:
            processed_data[col] = new_data[col]
        else:
            # For missing columns, add them with default values
            # For categorical columns that were encoded, use the most common value from training
            if col in label_encoders:
                most_common_value = train_data[col].mode()[0]
                processed_data[col] = most_common_value
            else:
                # For numerical columns, use the mean from training data
                processed_data[col] = train_data[col].mean()

    # Fill any remaining missing values with the mean of training data
    processed_data.fillna(train_data.mean(), inplace=True)

    # Verify that all expected columns are present
    assert set(processed_data.columns) == set(expected_columns), "Column mismatch after processing"

    # Apply the same preprocessing as during training if artifacts exist
    if scaler is not None and selector is not None and selected_features is not None:
        # Create a new DataFrame with just the selected features
        selected_data = pd.DataFrame()

        # For each selected feature, either copy it from processed_data or use a default value
        for col in selected_features:
            if col in processed_data.columns:
                selected_data[col] = processed_data[col]
            else:
                # Use the mean from training data for missing columns
                selected_data[col] = train_data[col].mean()

        # Fill any remaining missing values with the mean of training data
        selected_data.fillna(train_data.mean(), inplace=True)

        # Verify that all selected features are present
        assert set(selected_data.columns) == set(selected_features), "Selected feature mismatch after processing"

        # Make predictions directly using the selected features
        # No need to apply scaler or selector again as we're already using the selected features
        predictions = model.predict(selected_data)
        prediction_probs = model.predict_proba(selected_data)
    else:
        # Legacy prediction method
        predictions = model.predict(processed_data)
        prediction_probs = model.predict_proba(processed_data)

    # Convert numeric predictions to human-readable labels
    pred_labels = ['yes' if pred == 1 else 'no' for pred in predictions]

    return pred_labels, prediction_probs


if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog
    from tkinter import ttk


    def upload_file():
        """Let the user upload an Excel file and process it for predictions."""
        file_path = filedialog.askopenfilename(filetypes=[("Excel/CSV files", "*.xls;*.xlsx;*.csv")])
        if not file_path:
            return

        try:
            if file_path.endswith(('.xls', '.xlsx')):
                new_data = pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                new_data = pd.read_csv(file_path)
            else:
                raise ValueError("Unsupported file format. Please upload an Excel or CSV file.")

            predictions, probs = predict_new_data(new_data)

            for row in results_tree.get_children():
                results_tree.delete(row)

            for idx, (prediction, prob) in enumerate(zip(predictions, probs)):
                # Format probability as percentage
                prob_percent = f"{prob[1]*100:.2f}%"
                results_tree.insert("", "end", values=(idx + 1, prediction, prob_percent))

            messagebox.showinfo("Success", "Predictions completed successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process the file: {e}")


    def evaluate_current_model():
        """Evaluate the current model and show results."""
        try:
            metrics = evaluate_model()

            # Create a new window to display metrics
            metrics_window = tk.Toplevel(root)
            metrics_window.title("Model Evaluation Metrics")
            metrics_window.geometry("600x400")

            # Create a text widget to display metrics
            text_widget = tk.Text(metrics_window, wrap=tk.WORD, padx=10, pady=10)
            text_widget.pack(fill=tk.BOTH, expand=True)

            # Insert metrics
            text_widget.insert(tk.END, "MODEL EVALUATION METRICS\n\n")
            text_widget.insert(tk.END, f"Accuracy: {metrics['accuracy']:.4f}\n")

            if metrics['roc_auc']:
                text_widget.insert(tk.END, f"ROC AUC: {metrics['roc_auc']:.4f}\n")

            if metrics['pr_auc']:
                text_widget.insert(tk.END, f"PR AUC: {metrics['pr_auc']:.4f}\n")

            text_widget.insert(tk.END, f"Sensitivity: {metrics['sensitivity']:.4f}\n")
            text_widget.insert(tk.END, f"Specificity: {metrics['specificity']:.4f}\n\n")

            text_widget.insert(tk.END, "Confusion Matrix:\n")
            text_widget.insert(tk.END, f"{metrics['confusion_matrix']}\n\n")

            text_widget.insert(tk.END, "Classification Report:\n")
            text_widget.insert(tk.END, f"{metrics['classification_report']}\n")

            # Make text widget read-only
            text_widget.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to evaluate model: {e}")


    def retrain_model():
        """Retrain the model with current data."""
        try:
            if messagebox.askyesno("Confirm", "Are you sure you want to retrain the model? This may take some time."):
                # Show a progress dialog
                progress_window = tk.Toplevel(root)
                progress_window.title("Training Progress")
                progress_window.geometry("300x100")

                progress_label = tk.Label(progress_window, text="Training model... Please wait.")
                progress_label.pack(pady=20)

                # Update the UI
                progress_window.update()

                # Train the model
                global model, label_encoders
                model, X_train, X_test, y_train, y_test, label_encoders = train_new_model()

                # Close progress window
                progress_window.destroy()

                # Show success message
                messagebox.showinfo("Success", "Model retrained successfully!")

                # Evaluate the new model
                evaluate_current_model()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to retrain model: {e}")


    # Check if model exists or train a new one
    if os.path.exists(MODEL_FILENAME):
        print("Loading existing model...")
        model, train_data, test_data, label_encoders = load_model_artifacts()
        print(f"Model loaded. Training data shape: {train_data.shape}, Test data shape: {test_data.shape}")

        # Evaluate the model
        print("\nEvaluating model performance:")
        evaluate_model(model, test_data)
    else:
        print("Training new model...")
        model, X_train, X_test, y_train, y_test, label_encoders = train_new_model()

    # Create the main UI
    root = tk.Tk()
    root.title("Autism Diagnosis System")
    root.geometry("800x600")

    # Create a frame for buttons
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10, fill=tk.X)

    # Add buttons
    upload_btn = tk.Button(button_frame, text="Upload Data File", command=upload_file, width=20)
    upload_btn.pack(side=tk.LEFT, padx=10)

    evaluate_btn = tk.Button(button_frame, text="Evaluate Model", command=evaluate_current_model, width=20)
    evaluate_btn.pack(side=tk.LEFT, padx=10)

    # retrain_btn = tk.Button(button_frame, text="Retrain Model", command=retrain_model, width=20)
    # retrain_btn.pack(side=tk.LEFT, padx=10)

    # Create a frame for the results tree
    tree_frame = tk.Frame(root)
    tree_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    # Add a label
    results_label = tk.Label(tree_frame, text="Prediction Results:", font=("Arial", 12, "bold"))
    results_label.pack(anchor=tk.W, pady=(0, 5))

    # Create the results tree
    results_tree = ttk.Treeview(tree_frame, columns=("Row", "Prediction", "Probability"), show="headings")
    results_tree.heading("Row", text="Row")
    results_tree.heading("Prediction", text="Prediction")
    results_tree.heading("Probability", text="Probability")
    results_tree.column("Row", width=50)
    results_tree.column("Prediction", width=100)
    results_tree.column("Probability", width=100)

    # Add scrollbars
    vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=results_tree.yview)
    hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=results_tree.xview)
    results_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

    # Grid layout for tree and scrollbars
    results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    vsb.pack(side=tk.RIGHT, fill=tk.Y)
    hsb.pack(side=tk.BOTTOM, fill=tk.X)

    # Add status bar
    status_bar = tk.Label(root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    root.mainloop()
