# taoverse/model/storage/remote_model_store.py

from typing import Type, Tuple, Any, Dict
from temporal.utils.hf_accessors import save_hf, load_hf

class RemoteModelStore:
    def __init__(
        self,
        repo_id: str,
        token: str = None,
        safe_format: str = "pickle",
        commit_message: str = None,
    ):
        self.repo_id = repo_id
        self.token = token
        self.safe_format = safe_format
        self.commit_message = commit_message

    def save_model(
        self,
        model: Any,
        config: Any,
        **kwargs: Any  # accepts metagraph, model_constraints, etc.
    ) -> None:
        """
        Persist a model + its config to the HF repo.
        """
        save_hf(
            model=model,
            config=config,
            repo_id=self.repo_id,
            token=self.token,
            safe_format=self.safe_format,
            commit_message=self.commit_message,
        )

    def load_model(
        self,
        config_cls: Type,
        **kwargs: Any  # accepts metagraph, model_constraints, etc.
    ) -> Tuple[Any, Any]:
        """
        Load model + config from the HF repo.
        Returns (model, config_instance).
        """
        model, config_dict = load_hf(
            repo_id=self.repo_id,
            token=self.token,
            safe_format=self.safe_format,
        )
        if hasattr(config_cls, "from_dict"):
            config = config_cls.from_dict(config_dict)
        else:
            config = config_cls(**config_dict)  # fallback
        return model, config
