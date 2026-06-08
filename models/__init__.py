from .efficientb4 import EfficientB4Detector
from .fwa import FWADetector
from .spsl import SPSLDetector
from .srm import SRMDetector
from .ucf import UCFDetector


MODEL_REGISTRY = {
    "efficientb4": EfficientB4Detector,
    "fwa": FWADetector,
    "ucf": UCFDetector,
    "srm": SRMDetector,
    "spsl": SPSLDetector,
}


def build_model(model_name: str, config):
    key = model_name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model '{model_name}'. Choose from: {', '.join(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key](config)

