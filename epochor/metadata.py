# epochor/metadata.py

"""
Handles metadata tracking and model exchange information.

Wraps hypothetical ChainModelMetadataStore and HuggingFaceModelStore
to provide a unified interface for miners and validators.
"""

import time
from typing import Optional, Dict, Any

class ChainModelMetadataStore:
    """
    Hypothetical class to interact with a blockchain or distributed ledger
    for storing and retrieving model metadata.
    """
    def __init__(self):
        # Placeholder for actual connection/setup
        self.metadata_store = {} # uid -> list of metadata entries
        print("[INFO] ChainModelMetadataStore initialized (placeholder).")

    def store_metadata(self, uid: int, model_id: str, score: float, block_number: int, hf_repo_id: Optional[str] = None) -> bool:
        """
        Stores metadata associated with a model update.
        Returns True on success, False on failure.
        """
        if uid not in self.metadata_store:
            self.metadata_store[uid] = []
        
        entry = {
            "model_id": model_id, # Could be a hash or a specific version
            "hf_repo_id": hf_repo_id,
            "score": score,
            "block_number": block_number,
            "timestamp": time.time()
        }
        self.metadata_store[uid].append(entry)
        print(f"[INFO] Stored metadata for UID {uid}: {entry}")
        return True

    def read_metadata(self, uid: int) -> Optional[Dict[str, Any]]:
        """
        Reads the latest metadata for a given UID.
        Returns the metadata entry or None if not found.
        """
        if uid in self.metadata_store and self.metadata_store[uid]:
            latest_entry = max(self.metadata_store[uid], key=lambda x: x["block_number"])
            print(f"[INFO] Read latest metadata for UID {uid}: {latest_entry}")
            return latest_entry
        print(f"[INFO] No metadata found for UID {uid}")
        return None

    def get_last_updated_block(self, uid: int) -> Optional[int]:
        """
        Gets the block number of the last metadata update for a UID.
        """
        metadata = self.read_metadata(uid)
        if metadata:
            return metadata["block_number"]
        return None


class HuggingFaceModelStore:
    """
    Hypothetical class to interact with Hugging Face Hub
    for model storage and retrieval information.
    """
    def __init__(self):
        # Placeholder for actual HF API client
        print("[INFO] HuggingFaceModelStore initialized (placeholder).")

    def get_model_last_updated_timestamp(self, hf_repo_id: str) -> Optional[float]:
        """
        Retrieves the last updated timestamp for a model on Hugging Face.
        This would typically involve an API call to Hugging Face.
        """
        # Placeholder: In a real scenario, you'd use the HF API.
        # For now, returning None or a mock timestamp.
        print(f"[INFO] Queried HF for last update of {hf_repo_id} (placeholder).")
        # Example: return time.time() - random.randint(3600, 86400*5) # Mock a recent update
        return None # Placeholder: No actual HF call implemented


class EpochorMetadataStore:
    """
    Manages model metadata, combining on-chain/distributed ledger info
    with Hugging Face model details.
    """
    def __init__(self):
        self.chain_store = ChainModelMetadataStore()
        self.hf_store = HuggingFaceModelStore()
        print("[INFO] EpochorMetadataStore initialized.")

    def record_model_upload(self, uid: int, model_id: str, score: float, block_number: int, hf_repo_id: str) -> bool:
        """
        For a miner to record a new model upload with its associated score and HF repo.
        """
        # Potentially, first verify with hf_store that the model exists, or get commit hash etc.
        print(f"[INFO] Recording model upload for UID {uid}, Model ID {model_id}, HF Repo {hf_repo_id}, Score {score} at Block {block_number}")
        return self.chain_store.store_metadata(uid, model_id, score, block_number, hf_repo_id)

    def get_model_metadata(self, uid: int) -> Optional[Dict[str, Any]]:
        """
        For a validator to get the latest registered metadata for a miner's model.
        """
        print(f"[INFO] Getting model metadata for UID {uid}")
        return self.chain_store.read_metadata(uid)

    def get_last_updated_block(self, uid: int) -> Optional[int]:
        """
        For a validator to check the recency of a miner's model based on stored metadata.
        This refers to the block number when the metadata was recorded.
        """
        print(f"[INFO] Getting last updated block for UID {uid} from chain metadata")
        return self.chain_store.get_last_updated_block(uid)

    def check_hf_model_recency(self, hf_repo_id: str) -> Optional[float]:
        """
        For a validator to check how recently a model was updated on Hugging Face directly.
        (Could be used to cross-verify or independently check HF activity).
        """
        print(f"[INFO] Checking Hugging Face model recency for {hf_repo_id}")
        return self.hf_store.get_model_last_updated_timestamp(hf_repo_id)

# Example Usage (can be removed or kept for testing)
if __name__ == '__main__':
    metadata_store = EpochorMetadataStore()

    # Miner records an upload
    metadata_store.record_model_upload(uid=101, model_id="model_v1_hash", score=0.95, block_number=12345, hf_repo_id="user/repo101")
    metadata_store.record_model_upload(uid=102, model_id="model_x_hash", score=0.92, block_number=12350, hf_repo_id="user/repo102")
    metadata_store.record_model_upload(uid=101, model_id="model_v2_hash", score=0.98, block_number=12360, hf_repo_id="user/repo101")

    # Validator checks metadata
    meta_uid101 = metadata_store.get_model_metadata(uid=101)
    if meta_uid101:
        print(f"Validator sees for UID 101: Score {meta_uid101['score']} at block {meta_uid101['block_number']}, HF: {meta_uid101['hf_repo_id']}")

    last_block_uid101 = metadata_store.get_last_updated_block(uid=101)
    print(f"Validator: Last update block for UID 101 from chain metadata: {last_block_uid101}")

    last_block_uid103 = metadata_store.get_last_updated_block(uid=103) # Non-existent
    print(f"Validator: Last update block for UID 103 from chain metadata: {last_block_uid103}")

    # Validator checks HF recency (placeholder)
    hf_recency = metadata_store.check_hf_model_recency(hf_repo_id="user/repo101")
    print(f"Validator: HF model recency for user/repo101: {hf_recency}")
