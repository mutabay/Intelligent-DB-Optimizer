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
class LLMConfig:
    """LLM configuration."""
    # Default LLM provider
    default_provider: str = os.getenv("LLM_PROVIDER", "ollama")  # ollama, openai, simple
    
    # Ollama settings
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")  # Lightweight model
    
    # OpenAI settings (if needed)
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4")
    
    # Model parameters
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))  # Low for consistent SQL
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1000"))

@dataclass
class Config:
    """Main configuration class."""
    database: DatabaseConfig
    llm: LLMConfig
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.llm = LLMConfig()

# Global config instance
config = Config()