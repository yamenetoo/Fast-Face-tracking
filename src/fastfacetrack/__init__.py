"""fastfacetrack: fast face detection, extraction, target-face classification and evaluation on video."""
from .detectors import Detection, FaceDetector, HaarDetector, UltraLightDetector, make_detector
from .evaluation import combine_results, evaluate_video
from .extraction import FaceSet, extract_faces
from .models import MODEL_NAMES, FaceClassifier
from .pipeline import ExperimentConfig, run_experiment
from .simulation import IdentityBank, SimulationConfig, simulate_video
from .training import make_dataset, train_models

__version__ = "1.0.0"
__all__ = ["Detection", "FaceDetector", "HaarDetector", "UltraLightDetector", "make_detector", "evaluate_video",
           "combine_results", "FaceSet", "extract_faces", "MODEL_NAMES", "FaceClassifier", "ExperimentConfig",
           "run_experiment", "IdentityBank", "SimulationConfig", "simulate_video", "make_dataset", "train_models"]
