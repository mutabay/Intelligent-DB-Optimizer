"""
Configuration management with environment variables.
"""
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class DatabaseConfig:
    """Database configuration."""
    # PostgreSQL settings
    postgres_host: str = os.getenv("POSTGRES_HOST", "127.0.0.1")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5433"))
    postgres_user: str = os.getenv("POSTGRES_USER", "postgres")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "bayram")
    postgres_db: str = os.getenv("POSTGRES_DB", "postgres")
    
    # SQLite settings
    sqlite_db_path: str = os.getenv("SQLITE_DB_PATH", "test.db")
    
    # Default database type
    default_db_type: str = "sqlite"  # Can be "sqlite" or "postgresql"

@dataclass
class Config:
    """Main configuration class."""
    database: DatabaseConfig
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    def __init__(self):
        self.database = DatabaseConfig()

# Global config instance
config = Config()