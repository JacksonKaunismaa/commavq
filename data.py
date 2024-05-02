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
    """Dataset class for CommaVQ dataset. Loads and preprocesses videos from disk. Reads from disk are cached to speed up training.
    A given batch is created by picking a random video, and then randomly selecting a block of tokens (of length block_size) 
    from that video, which is put into the 'xy' key of the batch."""
    def __init__(self, dset, exp_cfg: ExperimentCfg, dset_cfg: CommaVQDatasetCfg, load_decoder=False):
        """
        dset: hf.Dataset or list of hf.Datasets containing the tokenized data
        exp_cfg: ExperimentCfg, which contains block size, batch_size, and whether its ddp or not
        dset_cfg: CommaVQDatasetCfg, which contains split_ranks and decoder_path
        load_decoder: bool, whether to load the CommaVQ VAE decoder. If False, the dataset will only return the tokenized data.
        """
        if isinstance(dset, hf.Dataset):
            self.dset = dset
        else:  # something like a list of hf.Datasets
            self.dset = hf.concatenate_datasets(dset)
        self.dset_len = len(self.dset)

        self.encoder = encoder.Encoder(dset_cfg.decoder_path, load_decoder=load_decoder)
        self.cfg = copy.copy(dset_cfg)
        self.exp_cfg = exp_cfg
        self.cache = {}  # surely this won't grow too large in memory
        
        if self.cfg.split_ranks and self.exp_cfg.ddp:
            self.dset = hf_distributed.split_dataset_by_node(self.dset, 
                                                             rank=utils.get_rank(), 
                                                             world_size=utils.get_world_size())

        self_dict = dict(self=self)
        self.dset = self.dset.to_iterable_dataset().map(self.load_sample, fn_kwargs=self_dict) \
                                                   .map(self.subsample, fn_kwargs=self_dict, remove_columns=['ids', 'path'])
        self.dset = self.dset.with_format("torch")


    def __len__(self):
        return self.dset_len

    @staticmethod
    def load_sample(example, self):  # adapted from commavq repo, we need have the arguments in this weird order because of .map()
        """Load a single video from disk (or from cache) and preprocess it by appending BOS and appending EOT. 
        Caches the loaded sample. Since the length of the video can be greater than block size, this doesn't return an actual 
        batch sample. Returns a dictionary with the key 'ids' containing the tokenized video."""
        if example['path'] not in self.cache:
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
        """Subsample a block of tokens from the full video. This is done by randomly selecting a starting index and 
        taking the next block_size tokens. If this would go past the end of the video, we wrap around to the beginning, 
        delimitted by an EOT token. Returns a dictionary with the key 'xy' containing the input and target tokens."""
        # can improve this by making wrapping circular (means distribution over tokens is uniform)
        # however, this does mean the model learns to repeat the same video after an EOS, but not really a big deal
        full_len = example['ids'].shape[0]
        start_idx = np.random.randint(0, full_len)
        if start_idx + self.exp_cfg.block_size + 1 > full_len:
            selection1 = example['ids'][start_idx:]
            selection2 = example['ids'][:self.exp_cfg.block_size+1-(full_len - start_idx)]
            selection = np.concatenate([selection1, selection2])
        else:
            selection = example['ids'][start_idx: start_idx + self.exp_cfg.block_size + 1]
        return {'xy': (selection[:-1], selection[1:])}
        

    def dataloader(self):
        """Create an infinite dataloader with pinned memory and persistent workers for the dataset."""
        rprint("created sampler",)
        return make_infinite(DataLoader(self.dset, batch_size=self.exp_cfg.batch_size,  # type: ignore
                          pin_memory=True,
                          num_workers=1,
                          persistent_workers=True))#self.cfg.num_workers//2)  #//3 since the cpus aren't enough???