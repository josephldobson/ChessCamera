import os
import sys

PLATFORM: str = sys.platform
NUM_PROCS: int = os.cpu_count()

PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))
FILES_PATH = os.path.join(PROJECT_PATH, "files")
IMAGES_PATH = os.path.join(FILES_PATH, "images")
CHESS_GAME_IMAGES_PATH = os.path.join(IMAGES_PATH, "chess_game")
DUCC_IMAGES_PATH = os.path.join(IMAGES_PATH, "ChessgameDUCC")
RANDOM_PICTURES_PATH = os.path.join(IMAGES_PATH, "random_pictures")
NPZS_PATH = os.path.join(FILES_PATH, "npzs")