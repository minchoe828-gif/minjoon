import sys
from loguru import logger
from pathlib import Path

def setup_project_logger(log_dir: str | Path):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()

    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}:{line}</cyan> | <level>{message}</level>",
        level='INFO'
    )

    logger.add(
        str(log_dir/ '{time:YYYY-MM-DD}.log'),
        format="{time:YYYY-MM-DD HH:mm:ss:SSS} | {level: <8} | {name}:{line} | {message}",
        level='DEBUG',
        rotation='10 MB',
        retention='30 days',
        enqueue=True,
    )
