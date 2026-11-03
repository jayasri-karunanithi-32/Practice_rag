import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

try:
    with psycopg.connect(DATABASE_URL) as connection:
        print("PostgreSQL Connected Successfully!")

except Exception as error:
    print("Connection Failed:", error)