"""supplemental code to alleviate porting tests from aiter"""

import os

AITER_TRITON_CONFIGS_PATH: str | None = None

if AITER_TRITON_CONFIGS_PATH is None:
  AITER_TRITON_CONFIGS_PATH = os.path.join(os.path.dirname(__file__), "aiter_configs")
