import logging
import sys

# ANSI escape codes for colors
COLORS = {
    "DEBUG": "\033[94m",  # Blue
    "INFO": "\033[92m",   # Green
    "WARNING": "\033[93m",  # Yellow
    "ERROR": "\033[91m",   # Red
    "RESET": "\033[0m",    # Reset
}

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        color = COLORS.get(record.levelname, COLORS["RESET"])
        reset = COLORS["RESET"]
        message = super().format(record)
        #return f"{color}{message}{reset}"
        return f"{message}"

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s"))

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.handlers = [console_handler]
