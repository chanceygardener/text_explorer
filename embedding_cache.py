import random
import collections
import hashlib
import os
import pickle
from typing import Union


class EmbeddingCacheManager:

    # Dataset files could be large
    # so take 64kb data chunks to
    # feed to md5
    BUF_SIZE = 65536

    def __init__(self, cache_dir=None):
        self.cache_dir = os.path.join(cache_dir, '.embedding_cache') \
            or os.path.realpath('.embedding_cache')
        cd = None
        if cache_dir:
            if not os.path.split(cache_dir)[-1] == '.embedding_cache':
                cd = os.path.join(cache_dir, '.embedding_cache')
        else:
            cd = os.path.realpath('.embedding_cache')
        self.cache_dir = cd

        self._init_cache()

    def _init_cache(self) -> None:
        if not os.path.isdir(self.cache_dir):
            os.mkdir(self.cache_dir)

    def _get_file_hash(self, fpaths, embedder_uri=None) -> str:
        hsh = hashlib.md5()
        if isinstance(fpaths, str):
            fpaths = [fpaths]
        for fpath in fpaths:
            with open(fpath, 'rb') as f:
                while True:
                    dat = f.read(self.BUF_SIZE)
                    if not dat:
                        break
                    hsh.update(dat)
        if embedder_uri:
            hsh.update(bytes(embedder_uri, encoding='utf-8'))
        return hsh.hexdigest()

    def cache_embeddings(self, fpaths, embeddings, embedder_url=None):
        file_hash = self._get_file_hash(fpaths, embedder_uri=embedder_url)
        with open(os.path.join(self.cache_dir, file_hash), 'wb') as ofile:
            pickle.dump(embeddings, ofile)

    def check_cache(self, fpaths, embedder_url=None):
        if isinstance(fpaths, str):
            fpaths = [fpaths]
        file_hash = self._get_file_hash(fpaths, embedder_uri=embedder_url)
        cached = None
        if file_hash in os.listdir(self.cache_dir):
            with open(os.path.join(self.cache_dir, file_hash), 'rb') as cfile:
                cached = pickle.load(cfile)
        return cached

    def get_size(self):
        total_bytes = 0
        for cache_file in os.listdir(self.cache_dir):
            total_bytes += os.path.getsize(
                os.path.join(self.cache_dir, cache_file))
        return total_bytes

    def list_caches(self):
        return os.listdir(self.cache_dir)

    def clear_cache(self):
        for cache_file in os.listdir(self.cache_dir):
            # this shouldn't be, but shouldn't cause a failure.
            if not os.path.isdir(cache_file):
                os.remove(os.path.join(self.cache_dir, cache_file))
