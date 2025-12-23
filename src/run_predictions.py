"""
This file executes the stored procedure to generate predictions.
"""

from src.snowflake_connection import get_snowpark_session

def execute_predictions():
    """
    Calls the stored procedure to generate churn predictions.
    """
    session = get_snowpark_session()
    
    print("🚀 Running churn predictions...")
    result = session.call('PREDICT_CHURN')
    print(f"✅ {result}")
    
    # Verify results
    predictions_df = session.table('CHURN_PREDICTIONS')
    print(f"\nTotal predictions: {predictions_df.count()}")
    
    # Show sample predictions
    print("\nSample predictions:")
    predictions_df.select(
        'CHURN',
        'PREDICTED_CHURN',
        'CHURN_PROBABILITY'
    ).show(10)
    
    # Show risk distribution
    print("\nRisk Category Distribution:")
    session.sql("""
        SELECT 
            RISK_CATEGORY,
            COUNT(*) as COUNT,
            ROUND(AVG(CHURN_PROBABILITY), 4) as AVG_PROBABILITY
        FROM CHURN_ANALYSIS_VIEW
        GROUP BY RISK_CATEGORY
        ORDER BY AVG_PROBABILITY DESC
    """).show()
    
    session.close()

if __name__ == "__main__":
    execute_predictions()
