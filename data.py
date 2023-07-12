import datasets as hf
import datasets.distributed as hf_distributed
import copy
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, DataLoader


from ..config_objects import ExperimentCfg, CommaVQDatasetCfg
from .. import utils
from ..utils import rprint
from . import encoder


def make_infinite(dataloader):
    while True:
        yield from dataloader

class CommaVQDataset():
    def __init__(self, dset, exp_cfg: ExperimentCfg, dset_cfg: CommaVQDatasetCfg, load_decoder=False):
        if isinstance(dset, hf.Dataset):
            self.dset = dset
        else:  # something like a list of hf.Datasets
            self.dset = hf.concatenate_datasets(dset)
        self.dset_len = len(self.dset)

        self.encoder = encoder.Encoder(dset_cfg.decoder_path, load_decoder=load_decoder)
        self.cfg = copy.copy(dset_cfg)
        self.exp_cfg = exp_cfg
        self.cache = {}  # surely this won't grow too large in memory
        
        # self.dset = self.dset.select(range(8))
        # rprint("pre split full data", len(self.dset), self.dset.data, "yeah")
        
        if self.cfg.split_ranks and self.exp_cfg.ddp:
            self.dset = hf_distributed.split_dataset_by_node(self.dset, 
                                                             rank=utils.get_rank(), 
                                                             world_size=utils.get_world_size())
        # rprint("post split full data", len(self.dset), self.dset.data, "haey")

        self_dict = dict(self=self)
        self.dset = self.dset.to_iterable_dataset().map(self.load_sample, fn_kwargs=self_dict) \
                                                   .map(self.subsample, fn_kwargs=self_dict, remove_columns=['ids', 'path'])
        self.dset = self.dset.with_format("torch")


    def __len__(self):
        return self.dset_len

    @staticmethod
    def load_sample(example, self):  # adapted from commavq repo
        if example['path'] not in self.cache:
            # rprint(self.cache.keys())
            tokens = np.load(example['path'])  # potentially add caching here, assuming hf doesn't already do that
            tokens = tokens.reshape(tokens.shape[0], -1)
            # prepend BOS_TOKEN
            tokens = np.c_[np.ones(len(tokens), dtype=np.int16)*self.encoder.bos_token, tokens]
            tokens = tokens.reshape(-1)
            # append EOT_TOKEN
            tokens = np.r_[tokens, self.encoder.eos_token]
            self.cache[example['path']] = {'ids': tokens.astype(np.int16)}
        else:
            pass
            # print("hit cache!")
        return self.cache[example['path']]
    
    @staticmethod  # this method has a flaw where EOT tokens are rarer than in the nanogpt/all concatened case
    def subsample(example, self): # definitely shouldn't cache this
        start_idx = np.random.randint(0, example['ids'].shape[0] - self.exp_cfg.block_size)
        # start_idx = np.random.randint(0, 3)
        selection = example['ids'][start_idx: start_idx + self.exp_cfg.block_size + 1]
        return {'xy': (selection[:-1], selection[1:])}
        

    def dataloader(self):
        # if self.exp_cfg.ddp:
        #     sampler = torch.utils.data.DistributedSampler(self, shuffle=True)
        # else:
        #     sampler = torch.utils.data.RandomSampler(self, replacement=False)
        print("created sampler", utils.get_rank())
        return make_infinite(DataLoader(self.dset, batch_size=self.exp_cfg.batch_size,  # type: ignore
                          pin_memory=True,
                          num_workers=1,
                          persistent_workers=True))#self.cfg.num_workers//2)  #//3 since the cpus aren't enough???