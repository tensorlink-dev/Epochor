import wandb
import logging
import os

logger = logging.getLogger(__name__)

class MetricsLogger:
    def __init__(self, project_name: str, entity: str = None, disabled: bool = False):
        """
        Initializes the WandB logger.

        Args:
            project_name: The name of the WandB project.
            entity: The WandB entity (team or user). Optional.
            disabled: If True, disables WandB logging (e.g., for local testing).
        """
        self.disabled = disabled
        if self.disabled:
            logger.info("WandB logging is disabled.")
            return

        if not os.getenv("WANDB_API_KEY"):
            logger.warning("WANDB_API_KEY not set. Disabling WandB logging.")
            self.disabled = True
            return

        try:
            wandb.init(project=project_name, entity=entity)
            logger.info(f"WandB initialized for project '{project_name}'" + (f", entity '{entity}'" if entity else ""))
        except Exception as e:
            logger.error(f"Failed to initialize WandB: {e}")
            self.disabled = True

    def log(self, data: dict, step: int = None):
        """
        Logs data to WandB.

        Args:
            data: A dictionary of metrics to log.
            step: The current step (e.g., block number). Optional.
        """
        if self.disabled:
            return
        try:
            if step is not None:
                wandb.log(data, step=step)
            else:
                wandb.log(data)
        except Exception as e:
            logger.error(f"Failed to log data to WandB: {e}")

    def finish(self):
        """
        Finishes the WandB run.
        """
        if not self.disabled:
            try:
                wandb.finish()
                logger.info("WandB run finished.")
            except Exception as e:
                logger.error(f"Failed to finish WandB run: {e}")

# Example usage (optional, can be removed or kept for testing)
if __name__ == '__main__':
    # Mock EPOCHOR_CONFIG for testing
    class MockConfig:
        WANDB_PROJECT = "epochor-test"
        WANDB_ENTITY = None # Or your entity

    EPOCHOR_CONFIG = MockConfig()

    # Ensure WANDB_API_KEY is set in your environment for this to run
    if os.getenv("WANDB_API_KEY"):
        metrics_logger = MetricsLogger(
            project_name=EPOCHOR_CONFIG.WANDB_PROJECT,
            entity=EPOCHOR_CONFIG.WANDB_ENTITY
        )
        metrics_logger.log({"test_metric": 1, "epoch": 1}, step=1)
        metrics_logger.log({"test_metric": 2, "epoch": 2}, step=2)
        metrics_logger.finish()
    else:
        print("Skipping MetricsLogger example: WANDB_API_KEY not set.")

    # Test disabled mode
    print("Testing disabled mode:")
    disabled_logger = MetricsLogger(project_name="test", disabled=True)
    disabled_logger.log({"wont_log": 1})
    disabled_logger.finish()
