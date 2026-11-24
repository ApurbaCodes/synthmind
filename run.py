"""
AI Assistant — Single entry point.

Usage:
    streamlit run run.py

Validates the environment, initializes all components in the correct
dependency order, and launches the Streamlit UI.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)


def _validate_environment() -> None:
    """Validate that all required configuration is present.

    Raises:
        SystemExit: If critical configuration is missing.
    """
    try:
        from config.settings import get_settings

        settings = get_settings()
        logger.info("Configuration validated: %s", settings)
    except Exception as exc:
        print(f"\n❌ Configuration Error: {exc}\n")
        print("Please check your .env file.  See .env.example for required variables.\n")
        sys.exit(1)


def main() -> None:
    """Application entry point."""
    _validate_environment()

    # Import and run the Streamlit UI
    from ui.app import main as run_ui

    run_ui()


if __name__ == "__main__":
    main()
