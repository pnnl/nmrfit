"""nmrfit reads the output from an NMR spectroscopy experiment and,
through a number of intuitive API calls, produces a least-squares
fit of Voigt-function approximations via particle swarm optimization.
Fitted peaks can then be used to perform quantitative NMR analysis,
including isotope ratio approximation."""

from .core import *
from . import plot
from . import containers
from . import equations
from . import utils
from . import proc_autophase

__version__ = '0.1.0'
