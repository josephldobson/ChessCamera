import os
import sys

from typing import *
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
import scipy

from tqdm import tqdm
from chesscamera.config.project_paths import *


class ChessCamera:
    CHESSBOARD_PHOTOS = [
        f"{os.path.join(CHESS_GAME_IMAGES_PATH, os.path.basename(i))}" 
        for i in os.listdir(CHESS_GAME_IMAGES_PATH)
    ]

    DUCC_PHOTOS = [
        f"{os.path.join(DUCC_IMAGES_PATH, os.path.basename(i))}"
        for i in os.listdir(DUCC_IMAGES_PATH)
    ]

    RANDOM_PHOTOS = [
        f"{os.path.join(RANDOM_PICTURES_PATH, os.path.basename(i))}"
        for i in os.listdir(RANDOM_PICTURES_PATH)
    ]

    TOTAL_IMAGES = len(CHESSBOARD_PHOTOS) + len(DUCC_PHOTOS) + len(RANDOM_PHOTOS)
    CHESSBOARD_PHOTO_NAMES = [os.path.basename(i) for i in CHESSBOARD_PHOTOS]
    DUCC_PHOTO_NAMES = [os.path.basename(i) for i in DUCC_PHOTOS]
    RANDOM_PHOTO_NAMES = [os.path.basename(i) for i in RANDOM_PHOTOS]

    ARGS = [i for i in dir() if "ARGS" not in i]

    def __init__(self, *args: Any, **kwargs: Union[AnyStr, Any]) -> None:
        self.args: tuple = args
        for k in kwargs.keys():
            if k in self.ARGS:
                setattr(self, k, kwargs[k])
            elif k not in self.ARGS:
                raise ValueError(f"Invalid Argument: {k}")
        
    @staticmethod
    def pack_npz(np_img_array, file_path):
        np.savez_compressed(file_path, *np_img_array)
        
    def main(self):
        img_dirs = []
        img_dirs.extend(self.CHESSBOARD_PHOTOS)
        img_dirs.extend(self.DUCC_PHOTOS)
        image_list = []
        for i in img_dirs:
            img = cv2.imread(i)
            try:
                image_list.append(np.asarray(img))
            except Exception as e:
                print(e)
                print(i)
                print(img)
                continue

        image_list = np.array(image_list)
        np.savez_compressed(os.path.join(NPZS_PATH, "photos.npz"), *image_list)
        print(image_list.shape)

if __name__ == "__main__":
    ChessCamera().main()
