from abc import ABC, abstractmethod

import numpy as np
import torch


class BindingTask(ABC):
    """
    Interface for a single binding task used in multi-task finetuning.

    To add a new task:
      1. Subclass BindingTask in finetune/binding/<task_name>.py
      2. Append an instance to TASKS in main.py
    """

    name: str             # used for dir naming, logging, checkpoint keys
    query_dim: int        # output dim of the frozen query encoder
    batch_size: int       # query mini-batch size during training
    test_split_path: str  # where to save/load the test split .pt file

    @abstractmethod
    def load_data(self) -> tuple:
        """
        Returns:
            mhc_tokens  : Tensor (n_mhc, seq_len) on CPU
            query_list  : list[str]  — all unique query strings, sorted
            train_idx   : np.ndarray
            val_idx     : np.ndarray
            test_idx    : np.ndarray
            train_pos   : dict[int, set[int]]  mhc_local → set of query indices
            val_pos     : dict[int, set[int]]
            test_pos    : dict[int, set[int]]
        """

    @abstractmethod
    def encode_queries(self, query_list: list[str]) -> torch.Tensor:
        """
        Encode all queries with a frozen external encoder, cache to disk.
        Returns (N, query_dim) float32 tensor on CPU.
        """
