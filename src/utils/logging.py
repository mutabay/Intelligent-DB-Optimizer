import logging
import sys

def setup_logging(level: str = "INFO"):
    """Set up basic logging."""
    
    # Create logger
    logger = logging.getLogger("db_optimizer")
    logger.setLevel(getattr(logging, level))
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    # Add handler to logger
    if not logger.handlers:
        logger.addHandler(handler)
    
    return logger

# Create default logger
logger = setup_logging()