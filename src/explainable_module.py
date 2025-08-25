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
        self.filepath = filepath

        if not os.path.exists(filepath):
            print(f"Cache for {filepath} not found, creating new cache.")
            self.cache = {}
        else:
            print(f"Cache for {filepath} found, loading cache.")
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
                "num_tokens":1
            }
        else:
            self.vectorizer_configs = configs

    def cache_features(self, uid, vector): 
        if uid not in self.cache:
            self.cache[uid] = vector

    def cache_vectors_batch(self, docs, uids):
        """Process multiple documents in a batch for better performance"""
        # Filter out documents that are already cached
        new_docs = []
        new_uids = []
        for doc, uid in zip(docs, uids):
            if uid not in self.cache:
                new_docs.append(doc)
                new_uids.append(uid)
        
        if not new_docs:
            return
        
        # Process the batch at once
        vectors = vectorizer.from_documents(new_docs, config=self.vectorizer_configs).values
        
        # Cache the results
        for uid, vector in zip(new_uids, vectors):
            self.cache_features(uid, vector)

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
        if normalized and not hasattr(self, 'normalized_cache'):
            print("normalized cache not found, creating normalized_cache")
            self.normalize_cache()
            cache = self.normalized_cache
        elif normalized and hasattr(self, 'normalized_cache'):
            if not hasattr(self, 'found_flag'):
                print("using normalized cache")
                self.found_flag = True
            cache = self.normalized_cache
        else:
            cache = self.cache

        # Better error handling
        if uid1 not in cache:
            raise KeyError(f"uid1 {uid1} not in cache")
        if uid2 not in cache:
            raise KeyError(f"uid2 {uid2} not in cache")
            
        vector_1 = cache[uid1].squeeze()
        vector_2 = cache[uid2].squeeze()
        
        # Reshape vectors to 2D arrays for cosine_similarity
        vector_1 = vector_1.reshape(1, -1)
        vector_2 = vector_2.reshape(1, -1)
        
        cosine_sim = cosine_similarity(vector_1, vector_2)[0][0]
        return vector_1, vector_2, cosine_sim

    def save_cache(self):
        with open(f"{self.filepath}", "wb") as f:
            pickle.dump(self.cache, f)

    def normalize_cache(self):
        start_time = time.time()
        self.normalized_cache = {}
        
        # Convert cache values to array and squeeze out extra dimension
        values_array = np.array(list(self.cache.values())).squeeze()
        
        # Calculate mean and std per feature
        vector_means = np.mean(values_array, axis=0)
        vector_stds = np.std(values_array, axis=0)
        
        # Handle zero standard deviations to avoid division by zero
        vector_stds[vector_stds == 0] = 1.0
        
        # Normalize each document's features independently
        for uid in self.cache:
            self.normalized_cache[uid] = (self.cache[uid].squeeze() - vector_means) / vector_stds
            
        print(f"Normalized cache in {time.time() - start_time:.2f} seconds")