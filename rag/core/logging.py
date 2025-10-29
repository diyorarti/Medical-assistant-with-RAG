import logging
import logging.config
import os

def init_logging():
    """
    Configure logging once at startup.
    Falls back to sane defaults if LOG_LEVEL not set.
    """
    level = os.getenv("LOG_LEVEL", "INFO").upper()

    LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            },
            "access": {
                "format": "%(asctime)s | %(levelname)s | %(client_addr)s - '%(request_line)s' %(status_code)s"
            },
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": level,
            },
        },
        "loggers": {
            "": {"handlers": ["default"], "level": level},
            # quiet noisy libs if needed:
            "uvicorn": {"level": "INFO"},
            "uvicorn.error": {"level": "INFO"},
            "uvicorn.access": {"level": "INFO"},
        },
    }

    logging.config.dictConfig(LOG_CONFIG)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
