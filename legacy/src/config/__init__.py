"""Configuration module for Genesis Sim."""
from .args import parse_args
from .init_sim import init_genesis

__all__ = ['parse_args', 'init_genesis']
