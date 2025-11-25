"""Checkpoint manager for saving and loading intermediate state."""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages checkpoints for resuming verification runs.

    Saves intermediate accumulators and processing state to allow
    resuming from failures or interruptions.
    """

    def __init__(self, checkpoint_dir: Path):
        """
        Initialize checkpoint manager.

        Parameters
        ----------
        checkpoint_dir : Path
            Directory for storing checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        checkpoint_id: str,
        data: Dict[str, Any],
    ) -> Path:
        """
        Save a checkpoint.

        Parameters
        ----------
        checkpoint_id : str
            Unique identifier for checkpoint
        data : Dict[str, Any]
            Data to checkpoint

        Returns
        -------
        Path
            Path to checkpoint file

        Examples
        --------
        >>> cm = CheckpointManager(Path("/output/checkpoints"))
        >>> cm.save_checkpoint("var_TMP_2m_00Z_f003", {"accumulators": accumulators})
        """
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.pkl"

        logger.info(f"Saving checkpoint: {checkpoint_id}")

        with open(checkpoint_file, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Checkpoint saved: {checkpoint_file}")
        return checkpoint_file

    def load_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint.

        Parameters
        ----------
        checkpoint_id : str
            Unique identifier for checkpoint

        Returns
        -------
        Optional[Dict[str, Any]]
            Checkpoint data, or None if not found
        """
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.pkl"

        if not checkpoint_file.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_id}")
            return None

        logger.info(f"Loading checkpoint: {checkpoint_id}")

        with open(checkpoint_file, "rb") as f:
            data = pickle.load(f)

        logger.info(f"Checkpoint loaded: {checkpoint_file}")
        return data

    def checkpoint_exists(self, checkpoint_id: str) -> bool:
        """
        Check if checkpoint exists.

        Parameters
        ----------
        checkpoint_id : str
            Unique identifier for checkpoint

        Returns
        -------
        bool
            True if checkpoint exists
        """
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.pkl"
        return checkpoint_file.exists()

    def list_checkpoints(self) -> list:
        """
        List all available checkpoints.

        Returns
        -------
        list
            List of checkpoint IDs
        """
        checkpoint_files = list(self.checkpoint_dir.glob("*.pkl"))
        return [f.stem for f in checkpoint_files]

    def delete_checkpoint(self, checkpoint_id: str) -> None:
        """
        Delete a checkpoint.

        Parameters
        ----------
        checkpoint_id : str
            Unique identifier for checkpoint
        """
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.pkl"

        if checkpoint_file.exists():
            checkpoint_file.unlink()
            logger.info(f"Deleted checkpoint: {checkpoint_id}")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_id}")

    def clear_all_checkpoints(self) -> None:
        """Delete all checkpoints."""
        checkpoint_files = list(self.checkpoint_dir.glob("*.pkl"))

        for checkpoint_file in checkpoint_files:
            checkpoint_file.unlink()

        logger.info(f"Cleared {len(checkpoint_files)} checkpoints")
