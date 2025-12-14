"""Slip metrics aggregation built atop SensorDataLogger."""
from typing import Dict, Any
from src.sensors.logger import SensorDataLogger


def summarize(logger: SensorDataLogger) -> Dict[str, Any]:
    return logger.get_summary_stats()


def detailed(logger: SensorDataLogger) -> Dict[str, Any]:
    return logger.get_slippage_metrics()
