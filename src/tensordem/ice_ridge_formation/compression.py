"""Compression driver: boundary-driven convergence of ice floes.

This module implements the mechanical driver for ice ridge formation:
two rigid walls converge at a constant velocity, compressing the ice floes
between them.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Tuple

import torch

__all__ = [
    "CompressionConfig",
    "CompressionDriver",
    "drive_compression",
]


@dataclass(frozen=True)
class CompressionConfig:
    """Configuration for the compression driver.
    
    Attributes
    ----------
    initial_gap : float
        Initial distance between the two walls (metres).
    compression_velocity : float
        Velocity of wall convergence (m/s).
    target_strain : float
        Target compressive strain (dimensionless).
    dt : float
        Time step (seconds).
    max_steps : int
        Maximum number of time steps.
    """
    initial_gap: float = 10.0
    compression_velocity: float = 0.5
    target_strain: float = 0.5
    dt: float = 1.0e-3
    max_steps: int = 10000
    
    def __post_init__(self) -> None:
        if self.initial_gap <= 0:
            raise ValueError("initial_gap must be positive")
        if self.compression_velocity <= 0:
            raise ValueError("compression_velocity must be positive")
        if not 0 < self.target_strain < 1:
            raise ValueError("target_strain must be in (0, 1)")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
    
    @property
    def target_displacement(self) -> float:
        """Target displacement of the walls (metres)."""
        return self.initial_gap * self.target_strain
    
    @property
    def target_time(self) -> float:
        """Time to reach target displacement (seconds)."""
        return self.target_displacement / self.compression_velocity


class CompressionDriver:
    """Driver for boundary-driven compression.
    
    The driver moves two rigid walls toward each other at a constant velocity,
    compressing the ice floes between them.
    
    Attributes
    ----------
    config : CompressionConfig
        Configuration for the compression.
    left_wall_x : float
        Current x-position of the left wall.
    right_wall_x : float
        Current x-position of the right wall.
    current_time : float
        Current simulation time.
    current_step : int
        Current time step number.
    """
    
    def __init__(self, config: CompressionConfig, domain_width: float):
        self.config = config
        self.domain_width = domain_width
        self.left_wall_x = 0.0
        self.right_wall_x = domain_width
        self.current_time = 0.0
        self.current_step = 0
        self.history: List[dict] = []
    
    @property
    def current_gap(self) -> float:
        """Current distance between the walls (metres)."""
        return self.right_wall_x - self.left_wall_x
    
    @property
    def current_strain(self) -> float:
        """Current compressive strain (dimensionless)."""
        displacement = self.config.initial_gap - self.current_gap
        return displacement / self.config.initial_gap
    
    @property
    def current_displacement(self) -> float:
        """Current displacement of the walls (metres)."""
        return self.config.initial_gap - self.current_gap
    
    def step(self) -> bool:
        """Advance the compression by one time step.
        
        Returns
        -------
        bool
            True if the target strain has been reached, False otherwise.
        """
        if self.current_step >= self.config.max_steps:
            return True
        
        # Move walls
        half_velocity = 0.5 * self.config.compression_velocity
        self.left_wall_x += half_velocity * self.config.dt
        self.right_wall_x -= half_velocity * self.config.dt
        
        self.current_time += self.config.dt
        self.current_step += 1
        
        # Record history
        self.history.append({
            "time": self.current_time,
            "step": self.current_step,
            "gap": self.current_gap,
            "strain": self.current_strain,
            "displacement": self.current_displacement,
            "left_wall_x": self.left_wall_x,
            "right_wall_x": self.right_wall_x,
        })
        
        # Check if target reached
        return self.current_strain >= self.config.target_strain
    
    def run(self) -> dict:
        """Run the compression until target strain is reached.
        
        Returns
        -------
        dict
            Summary of the compression history.
        """
        while not self.step():
            pass
        
        return {
            "final_time": self.current_time,
            "final_step": self.current_step,
            "final_gap": self.current_gap,
            "final_strain": self.current_strain,
            "final_displacement": self.current_displacement,
            "history": self.history,
        }


def drive_compression(
    config: CompressionConfig,
    domain_width: float,
) -> CompressionDriver:
    """Convenience function to run a compression simulation.
    
    Parameters
    ----------
    config : CompressionConfig
        Configuration for the compression.
    domain_width : float
        Initial width of the domain (metres).
    
    Returns
    -------
    CompressionDriver
        The compression driver with completed history.
    """
    driver = CompressionDriver(config, domain_width)
    driver.run()
    return driver
