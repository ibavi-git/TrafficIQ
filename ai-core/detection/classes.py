"""COCO vehicle classes used for TrafficIQ M1–M3.

bicycle + motorcycle are both counted as "bike".
"""

from typing import Dict, Optional, Tuple

# COCO ids: bicycle, car, motorcycle, bus, truck
COCO_VEHICLE_IDS: Tuple[int, ...] = (1, 2, 3, 5, 7)

CLASS_TO_GROUP: Dict[str, str] = {
    "bicycle": "bike",
    "motorcycle": "bike",
    "car": "car",
    "bus": "bus",
    "truck": "truck",
}

COUNT_GROUPS: Tuple[str, ...] = ("car", "bike", "bus", "truck")

DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_DEVICE = "cpu"
DEFAULT_CONF = 0.25


def group_for_class(name: str) -> Optional[str]:
    return CLASS_TO_GROUP.get(name.lower())
