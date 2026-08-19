"""Init for ReadSam."""
from __future__ import absolute_import
from .read_IDAT import read_idat
from .RawArray import RawArray, read_all_idats, find_files, find_files_v2, find_files_v3, multiprocess_read_all_idats, validate_idat_pairs
