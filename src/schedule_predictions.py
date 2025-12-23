"""
This file sets up automatic prediction runs using Snowflake tasks.
"""

from src.snowflake_connection import get_snowpark_session

def create_scheduled_task():
    """
    Creates a Snowflake task that runs predictions daily.
    """
    session = get_snowpark_session()
    
    # Create task
    create_task_sql = """
    CREATE OR REPLACE TASK DAILY_CHURN_PREDICTION
    WAREHOUSE = COMPUTE_WH
    SCHEDULE = 'USING CRON 0 2 * * * UTC'  -- Runs daily at 2 AM UTC
    AS
    CALL PREDICT_CHURN();
    """
    
    session.sql(create_task_sql).collect()
    print("✅ Task 'DAILY_CHURN_PREDICTION' created!")
    
    # Resume task (tasks are created in suspended state)
    session.sql("ALTER TASK DAILY_CHURN_PREDICTION RESUME").collect()
    print("✅ Task activated and will run daily at 2 AM UTC")
    
    session.close()

if __name__ == "__main__":
    create_scheduled_task()
