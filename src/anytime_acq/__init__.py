"""Amortized acquisition control for Bayesian optimization: does switching between acquisition functions,
conditioned on a frozen PFN's in-context belief, pay off, and how much could a policy amortized over the prior
gain?"""

from importlib.metadata import version

__version__ = version("anytime-acquisition")
