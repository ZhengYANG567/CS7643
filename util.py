import os, sys
import argparse
import logging
import hashlib

# Colors
GRAY = '\033[90m'
RED = '\033[91m'
BLUE = '\033[94m'
BOLD = '\033[1m'
RESET = '\033[0m'

# ANSI escape codes for colors
DEBUG_COLOR = '\033[90m'
INFO_COLOR = '\033[92m'
WARNING_COLOR = '\033[93m'
ERROR_COLOR = '\033[91m'
CRITICAL_COLOR = '\033[101;1m'
RESET = '\033[0m'

class CustomFormatter(logging.Formatter):
    def __init__(self, show_location=False):
        self.show_location = show_location
        super().__init__()

    def format(self, record):
        # Color handling based on log level
        color = ""
        if record.levelno == logging.DEBUG:
            color = DEBUG_COLOR if sys.stdout.isatty() else ""
        elif record.levelno == logging.INFO:
            color = INFO_COLOR if sys.stdout.isatty() else ""
        elif record.levelno == logging.WARNING:
            color = WARNING_COLOR if sys.stderr.isatty() else ""
        elif record.levelno == logging.ERROR:
            color = ERROR_COLOR if sys.stderr.isatty() else ""
        elif record.levelno == logging.CRITICAL:
            color = CRITICAL_COLOR if sys.stderr.isatty() else ""

        # Log format with optional location (file name and line number)
        location = f" ({record.filename}:{record.lineno})" if self.show_location else ""
        msg_format = f"{color}[{record.levelname}]{location} {record.getMessage()}{location}{RESET}"

        return msg_format

class CustomLogger:
    def __init__(self, name, show_location=False):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Clear previous handlers to avoid duplicate logging
        self.logger.handlers.clear()

        # Stream handler for stdout (DEBUG and INFO)
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)
        stdout_handler.setFormatter(CustomFormatter(show_location))

        # Stream handler for stderr (ERROR and CRITICAL)
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.ERROR)
        stderr_handler.setFormatter(CustomFormatter(show_location))

        self.logger.addHandler(stdout_handler)
        self.logger.addHandler(stderr_handler)

    def get_logger(self):
        return self.logger

def setup_logger(logger = logging.getLogger(), verbose=False, filename=False):
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Clear previous handlers to avoid duplicate logging
    logger.handlers.clear()

    # Stream handler for stdout (DEBUG and INFO)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.addFilter(lambda record: record.levelno <= logging.WARNING)
    stdout_handler.setFormatter(CustomFormatter(filename))

    # Stream handler for stderr (ERROR and CRITICAL)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.ERROR)
    stderr_handler.setFormatter(CustomFormatter(filename))

    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)

def set_fig_style(xlabel="", ylabel="", title=""):
    import matplotlib as mpl
    # mpl.rcParams["font.size"] = 20
    mpl.rcParams["axes.labelsize"] = 20
    mpl.rcParams["axes.titlesize"] = 24
    import matplotlib.pyplot as plt
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    plt.title(title, fontsize=30)
    plt.tight_layout()

def generate_checksum(filepath, algorithm='md5'):
    """Generates a checksum for a file.

    Args:
        filepath: The path to the file.
        algorithm: The hashing algorithm to use (e.g., 'md5', 'sha256').

    Returns:
        The checksum as a hexadecimal string, or None if an error occurs.
    """
    try:
        hasher = hashlib.new(algorithm)
        with open(filepath, 'rb') as file:
            while True:
                chunk = file.read(4096)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        print(f"Error generating checksum: {e}")
        return None

class CustomHelpFormatter(argparse.HelpFormatter):
    def _get_help_string(self, action):
        help_msg = action.help
        if action.required:
            help_msg += f" {RED}(required){RESET}"
        if action.default is not argparse.SUPPRESS and action.default is not None:
            help_msg += f" {GRAY}(default: {action.default}){RESET}"
        return help_msg

    def start_section(self, heading):
        if heading == 'positional arguments' or heading == 'optional arguments':
            super().start_section(heading)
        else:
            super().start_section(f"{BOLD}{BLUE}{heading}{RESET}")

def get_parser(add_verbose=False):
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    parser.add_argument("--verbose", "-v", action="store_true",
        help = "Increase verbosity level")
    return parser
