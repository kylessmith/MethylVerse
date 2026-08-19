from __future__ import absolute_import
from . import core
from .core.methyl_core import read_methylation
from .core.microarray.read_IDAT import read_idat
from .core.microarray.RawArray import RawArray
from .core.sequencing.sequencing_core import process_sequencing
from .core.utilities import *
from .plot.plot_plt import *
from .tools.decomposition.decompose import *
from multiverse_cache import download_dataset

from . import tools as tl
from . import plot as pl
from . import recipes

# Download annotation data
download_dataset("methyl_anno")

# This is extracted automatically by the top-level setup.py.
__version__ = '1.3.0'
__author__ = "Kyle S. Smith"


__doc__ = """\
API
======

Basic class
-----------

.. autosummary::
   :toctree: .
   
   read_idat
    
"""