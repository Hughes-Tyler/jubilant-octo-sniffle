import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve username and password from environment variables
USERNAME = os.getenv('STREAMLIT_USERNAME')
PASSWORD = os.getenv('STREAMLIT_PASSWORD')

def main():
    # Check login state
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.title("Login Page")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username == USERNAME and password == PASSWORD:
                st.session_state.logged_in = True
            else:
                st.error("Invalid username or password")
    else:
        run_ml_app()

def run_ml_app():
    st.title("Machine Learning Prediction App")

    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("Data Overview")
        st.write("Displaying first few rows of the dataset:")
        st.write(df.head())

        # Add preprocessing steps here
        df['QuoteStatus'] = df['Quote Status Code'].apply(lambda x: 1 if x == 'Ordered' else 0)
        feature_columns = ['Quote Id','Brand Code', 'Line Number', 'Product Group', 'Product Group Code',
                           'Product Summary Code','Product Summary', 'Product Name', 'Product Code',  'Pricing Auto Approval',
                           'Project Territory Name', 'Project Territory Code','Vertical Market Code', 'Distributor Name',
                           'Distributor Number', 'Created Date', 'Total Net Net',
                           'Total Unit Price', 'Discount', 'BookNetExt']
        X = df[feature_columns]
        y = df['QuoteStatus']

        X_categorical = X.select_dtypes(include=['object'])
        for col in X_categorical.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

        st.write("Displaying first few rows after preprocessing:")
        st.write(X.head())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Display metrics
        st.subheader("Model Performance Metrics")
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        st.write(f'Accuracy: {accuracy:.2f}')
        st.write(f'ROC AUC: {roc_auc:.2f}')

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        st.write(f'Precision: {precision:.2f}')
        st.write(f'Recall: {recall:.2f}')
        st.write(f'F1 Score: {f1:.2f}')

        st.subheader("Classification Report")
        class_report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(class_report).transpose())

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        conf_matrix = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, 
                    xticklabels=["Predicted Negative", "Predicted Positive"], 
                    yticklabels=["Actual Negative", "Actual Positive"])
        plt.xlabel("Predicted Labels")
        plt.ylabel("Actual Labels")
        plt.title("Confusion Matrix")
        st.pyplot(fig)

        # Feature Importances
        st.subheader("Feature Importances")
        feature_importances = model.feature_importances_
        feature_importances_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        st.dataframe(feature_importances_df)

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.barh(feature_importances_df['Feature'], feature_importances_df['Importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importances')
        plt.gca().invert_yaxis()
        st.pyplot(fig)

if __name__ == '__main__':
    main()
