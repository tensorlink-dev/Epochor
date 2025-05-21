# epochor/logging.py

import logging
import sys

# Store the initial logging configuration to allow reinitialization
_initial_log_level = logging.INFO
_log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Keep track of whether basicConfig has been called by this module
_configured = False

def reinitialize(log_level: int = None, force_reinit: bool = False) -> logging.Logger:
    """
    Configures and returns a logger for the 'epochor' namespace.
    It ensures that basicConfig is called only once unless forced.

    Args:
        log_level: The logging level (e.g., logging.INFO, logging.DEBUG).
                   Defaults to the module's _initial_log_level.
        force_reinit: If True, forces re-configuration even if already configured.
                      This is generally not recommended if other parts of the
                      application also configure logging.

    Returns:
        A logging.Logger instance.
    """
    global _configured, _initial_log_level

    chosen_log_level = log_level if log_level is not None else _initial_log_level

    # Python's logging.basicConfig() can only be effectively called once.
    # Subsequent calls are no-ops unless force=True is used (Python 3.8+).
    # For broader compatibility and to avoid issues with other modules configuring logging,
    # we try to be careful here.
    
    # If force_reinit, and Python version supports it, try to remove handlers and reconfigure.
    # This is a bit more involved and can have side effects.
    # A simpler approach is to just set the level of the root logger or specific loggers.

    if not _configured or force_reinit:
        # For force_reinit, if using Python 3.8+, can do:
        # logging.basicConfig(level=chosen_log_level, format=_log_format, force=True)
        # However, 'force' is not available in older Pythons.
        # A more compatible way if re-configuring: remove existing handlers.
        if force_reinit:
            # This removes handlers from the root logger.
            # Be cautious if other libraries add handlers you want to keep.
            root = logging.getLogger()
            for handler in root.handlers[:]:
                root.removeHandler(handler)
            logging.basicConfig(level=chosen_log_level, format=_log_format, stream=sys.stdout)
        else:
             logging.basicConfig(level=chosen_log_level, format=_log_format, stream=sys.stdout)
        
        _initial_log_level = chosen_log_level # Store the level that was used
        _configured = True
        logging.getLogger(__name__).info(f"Logging reinitialized to level {logging.getLevelName(chosen_log_level)}")

    # Get a logger specific to the 'epochor' namespace or the calling module's namespace.
    # Using 'epochor' as a root namespace for the project is a good practice.
    logger = logging.getLogger("epochor")
    logger.setLevel(chosen_log_level) # Ensure this specific logger instance has the desired level
    
    # If no handlers are configured for this logger specifically, and basicConfig was called,
    # it will propagate to the root logger's handlers.
    # If you want specific handlers (e.g., file output) for 'epochor', add them here.

    return logger

# Initialize once with default settings when module is imported
# if not _configured:
#    reinitialize() # Or defer to first explicit call

# Default logger for simple import-and-use cases, configured on first call to reinitialize or import
# logger = logging.getLogger("epochor") # This would get a logger that's not yet configured by reinitialize
