import requests
import os.path as osp
import numpy as np
import regex as re  # required since regular re does not support character classes \p
import multiprocessing as mp
import itertools
from tqdm import tqdm
import glob
import json
from typing import List
import onnxruntime as ort

from .utils import video  # yeah this is kinda ugly having 2 different things called utils
from ..utils import get_rank, get_device_type


class Encoder():  # or should it be called tokenizer
    def __init__(self, decoder_path, load_decoder=False):
        self.bos_token = 1024  # can hardcode these since CommaVQ is fixed to bit width of 10
        self.eos_token = 1025
        if load_decoder and get_rank() == 0:
            provider = f"{get_device_type().upper()}ExecutionProvider"
            print("provider is", provider)
            self.sess = ort.InferenceSession(decoder_path, 
                                             ort.SessionOptions(), 
                                             [provider])



    def decode(self, idx_list: np.ndarray):
        if not hasattr(self, "sess"):
            raise ValueError("Attempting to decode with a dummy encoder (ie. load_decoder=False).")
        pred_frames = []
        for frame in tqdm(idx_list.astype(np.int64)):
            preds = self.sess.run(None, {"encoding_indices": frame[None,...].reshape(1,8,16)})
            pred_frames.append(preds[0])
        return video.transpose_and_clip(pred_frames)
        