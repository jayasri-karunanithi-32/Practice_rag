import asyncpg
import os
from dotenv import load_dotenv
from pgvector.asyncpg import register_vector

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

async def init_connection(conn):
    await register_vector(conn)

async def connect_db():
    return await asyncpg.create_pool(
        DATABASE_URL,
        init=init_connection
    )