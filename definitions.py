import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "checkpoints")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
TRAIN_LOGS_DIR = os.path.join(ROOT_DIR, "training-logs")
# todo: add entrez config path (https://stackoverflow.com/questions/25389095/python-get-path-of-root-project-structure)
