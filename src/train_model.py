"""
This file trains a churn prediction model using data from Snowflake.
It uses SnowPark to read data, trains a model locally, and saves it.
"""

from src.snowflake_connection import get_snowpark_session
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import pandas as pd


def train_churn_model():
    """
    Trains a Random Forest model for churn prediction.
    """

    # Create Snowflake session
    session = get_snowpark_session()

    # Read data from Snowflake using SnowPark
    print("📊 Reading data from Snowflake...")
    df_snow = session.table("TELECOM_CHURN_V")


    # Convert SnowPark DataFrame to Pandas
    df = df_snow.to_pandas()
    print(f"Data shape: {df.shape}")

    # ---------------------------------------------------
    # Normalize column names (important for Snowflake)
    # ---------------------------------------------------
    # Normalize column names to lowercase (CRITICAL)
    df.columns = df.columns.str.lower()

# Debug print (run once)
    print("Available columns:")
    print(df.columns.tolist())

# Ensure churn column exists
    if 'churn' not in df.columns:
        raise ValueError("❌ 'churn' column not found after normalization")

# Prepare features and target
    exclude_cols = ['churn', 'customerid']

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df.drop(columns=["churn"])
    y = df["churn"].astype(int)


    # ---------------------------------------------------
    # Train-test split
    # ---------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ---------------------------------------------------
    # Train Random Forest model
    # ---------------------------------------------------
    print("\n🤖 Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # ---------------------------------------------------
    # Evaluate model
    # ---------------------------------------------------
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n✅ Model trained successfully!")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ---------------------------------------------------
    # Feature importance
    # ---------------------------------------------------
    feature_importance = (
        pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        })
        .sort_values('importance', ascending=False)
    )

    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))

    # ---------------------------------------------------
    # Save model and metadata
    # ---------------------------------------------------
    joblib.dump(model, 'churn_model.joblib')
    joblib.dump(feature_cols, 'feature_columns.joblib')

    print("\n💾 Model saved as 'churn_model.joblib'")
    print("💾 Feature columns saved as 'feature_columns.joblib'")

    # Close Snowflake session
    session.close()

    return model, feature_cols, accuracy


if __name__ == "__main__":
    train_churn_model()
