"""Shared utilities."""

from gen_sci.utils.device import get_device
from gen_sci.utils.seed import config_hash, set_seed, set_torch_seed

__all__ = ["config_hash", "get_device", "set_seed", "set_torch_seed"]
