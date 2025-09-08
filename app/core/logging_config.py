import logging
import sys
from pathlib import Path

# Ensure logs directory exists
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # change to DEBUG for more verbosity
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # keep printing to console
        logging.FileHandler(log_dir / "app.log",
                            encoding="utf-8")  # log to file
    ]
)

logger = logging.getLogger("app")
