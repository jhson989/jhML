
r"""
jhML : A Useful Deep Learning Framework Implemented by jhson
jhML framework is originated from "dezero" project, and some 
more useful utilities will be added. Feel free to use this framework to study
concepts of modern deep learning frameworks and a dynamic graph.
jhML contains data structures for multi-dimensional tensors 
and defines mathematical operations over these tensors.
Additionally, it provides utilities for developing deep learning applications

It will support GPGPU features, so you can run your tensor computations on an GPU someday. :)
"""

from jhML.core import ProgramConfig
from jhML.core import Variable
from jhML.core import Function
from jhML.core import tensor
from jhML.core import setup_variable
from jhML.utils.data import Dataset, Dataloader
from jhML.compute import as_cpu, as_gpu, get_array_module
import numpy as np



setup_variable()
__version__ = "0.0.1"

