import logging

_default_log_format = '[%(levelname)s] %(name)s at %(asctime)s --- %(message)s'
logging.basicConfig(format=_default_log_format, level=logging.WARNING)
root_logger = logging.getLogger()

from test_functions.function_realworld_bo.hpobench.__version__ import __version__  # noqa: F401, E402
from test_functions.function_realworld_bo.hpobench.config import config_file  # noqa: F401, E402

__contact__ = "automl.org"
