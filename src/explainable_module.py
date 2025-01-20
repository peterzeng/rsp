from gram2vec import vectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import time
import os

'''
    This module is used to get the vector representation of a document and the cosine similarity between two documents.
    We assume the cache is not normalized unless otherwise specified.
'''
class ExplainableModule():
    def __init__(self) -> None:
        pass

    def get_vector_and_score(self, doc1, doc2):
        pass

class Gram2VecModule(ExplainableModule):
    def __init__(self, filepath, dataset, save_dir, run_id, configs) -> None:
        super().__init__()
        self.dataset = dataset
        self.save_dir = save_dir
        self.run_id = run_id

        if not os.path.exists(filepath):
            print(f"Cache for {filepath} not found, creating new cache.")
            self.cache = {}
        else:
            with open(filepath, "rb") as f:
                self.cache = pickle.load(f)

        if not configs:
            self.vectorizer_configs = {
                "pos_unigrams":1,
                "pos_bigrams":1,
                "func_words":1,
                "punctuation":1,
                "letters":0,
                "emojis":1,
                "dep_labels":1,
                "morph_tags":1,
                "sentences":1,
                "num_tokens":0
            }
        else:
            self.vectorizer_configs = configs

    def cache_features(self, uid, vector): 
        if uid not in self.cache:
            self.cache[uid] = vector

    def cache_vector(self, doc, uid):
        if uid not in self.cache:
            vector = vectorizer.from_documents([doc], config = self.vectorizer_configs).values
            self.cache_features(uid, vector)

    def get_vector(self, doc, uid):
        if uid in self.cache:
            return self.cache[uid]
        else:
            vector = vectorizer.from_documents([doc], config = self.vectorizer_configs).values
            self.cache_features(uid, vector)
            return vector

    # Can only be ran after vector cache is created.
    def get_vector_and_score(self, doc1, doc2, uid1='', uid2='', normalized=False):
        if normalized and not self.normalized_cache:
            print("normalized cache not found, creating normalized_cache")
            self.normalize_cache()
            cache = self.normalized_cache
        else:
            cache = self.cache

        try:
            vector_1 = cache[uid1]
        except KeyError:
            print(f"uid1 {uid1} not in cache")

        try:
            vector_2 = cache[uid2]
        except KeyError:
            print(f"uid2 {uid2} not in cache")

        cosine_sim = cosine_similarity(vector_1, vector_2)[0][0]
        return vector_1, vector_2, cosine_sim

    def save_cache(self):
        with open(f"{self.save_dir}/{self.dataset}_{self.run_id}_vector_map.pkl", "wb") as f:
            pickle.dump(self.cache, f)

    def normalize_cache(self):
        start_time = time.time()
        self.normalized_cache = {}
        vector_std = np.std(list(self.cache.values()))
        vector_mean = np.mean(list(self.cache.values()))
        for uid in self.cache:
            self.normalized_cache[uid] = (self.cache[uid] - vector_mean) / vector_std
        print(f"Normalized cache in {time.time() - start_time:.2f} seconds")
