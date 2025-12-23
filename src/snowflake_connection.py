"""
This file handles the connection to Snowflake.
It reads credentials from .env file and creates a session.
"""

from snowflake.snowpark import Session
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def get_snowpark_session():
    """
    Creates and returns a Snowflake Snowpark session.
    This session will be used to run queries and operations in Snowflake.
    """
    connection_parameters = {
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "user": os.getenv("SNOWFLAKE_USER"),
        "password": os.getenv("SNOWFLAKE_PASSWORD"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
        "database": os.getenv("SNOWFLAKE_DATABASE"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA"),
        "role": os.getenv("SNOWFLAKE_ROLE")
    }
    
    session = Session.builder.configs(connection_parameters).create()
    print("✅ Connected to Snowflake successfully!")
    return session

# Test connection
if __name__ == "__main__":
    session = get_snowpark_session()
    print(f"Current database: {session.get_current_database()}")
    print(f"Current schema: {session.get_current_schema()}")
    session.close()
