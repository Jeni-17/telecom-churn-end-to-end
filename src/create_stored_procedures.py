"""
This file creates Snowflake stored procedures using SnowPark.
These procedures will run prediction inside Snowflake.
"""

from src.snowflake_connection import get_snowpark_session
import joblib


def deploy_prediction_procedure():
    """
    Creates a stored procedure in Snowflake that predicts churn.
    """
    session = get_snowpark_session()

    # Upload model artifacts to Snowflake stage
    print("📤 Uploading model to Snowflake stage...")
    session.file.put(
        "churn_model.joblib",
        "@CHURN_STAGE",
        auto_compress=False,
        overwrite=True
    )

    session.file.put(
        "feature_columns.joblib",
        "@CHURN_STAGE",
        auto_compress=False,
        overwrite=True
    )

    # SQL to create stored procedure
    create_proc_sql = """
    CREATE OR REPLACE PROCEDURE PREDICT_CHURN()
    RETURNS STRING
    LANGUAGE PYTHON
    RUNTIME_VERSION = '3.10'
    PACKAGES = ('snowflake-snowpark-python', 'scikit-learn', 'joblib', 'pandas')
    IMPORTS = ('@CHURN_STAGE/churn_model.joblib', '@CHURN_STAGE/feature_columns.joblib')
    HANDLER = 'predict_churn'
    AS
    $$
import joblib
import sys
from snowflake.snowpark import Session

def predict_churn(session: Session):

    # Load model files from Snowflake stage
    import_dir = sys._xoptions.get("snowflake_import_directory")
    model = joblib.load(import_dir + 'churn_model.joblib')
    feature_cols = joblib.load(import_dir + 'feature_columns.joblib')

    # Read source data
    df = session.table('TELECOM_CHURN').to_pandas()

    # Prepare features
    X = df[feature_cols]

    # Predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    # Add results
    df['PREDICTED_CHURN'] = predictions
    df['CHURN_PROBABILITY'] = probabilities

    # Save predictions
    result_df = session.create_dataframe(df)
    result_df.write.mode('overwrite').save_as_table('CHURN_PREDICTIONS')

    return f"Predictions completed successfully for {len(df)} records."
    $$;
    """

    print("🔧 Creating stored procedure in Snowflake...")
    session.sql(create_proc_sql).collect()
    print("✅ Stored procedure 'PREDICT_CHURN' created successfully!")

    session.close()


if __name__ == "__main__":
    deploy_prediction_procedure()
