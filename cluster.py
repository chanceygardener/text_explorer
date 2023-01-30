# -*- coding: utf-8 -*-
from embedding_cache import EmbeddingCacheManager
import pickle as pk
import argparse as ap
import tensorflow as tf
import tensorflow_text as text
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, space_eval, Trials
import umap
from sentence_transformers import SentenceTransformer
from sklearn.cluster import OPTICS, DBSCAN, AffinityPropagation, BisectingKMeans, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_samples, silhouette_score
from functools import partial
import hdbscan
import tensorflow_hub as hub
from pretrained_url import map_model_to_preprocess, map_name_to_handle
import numpy as np
import sys
from datetime import datetime
from tqdm import tqdm
import logging
import json
import re
from typing import Tuple
from uuid import uuid4 as uuid
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

logging.captureWarnings(True)
parser = ap.ArgumentParser('cluster')
parser.add_argument('input_path', type=str)
parser.add_argument('--output', '-o', type=str, default=os.getcwd())
parser.add_argument('--scoring', '-s', default='probability-threshold',
                    choices=['probability-threshold', 'avg-silhouette'])
parser.add_argument('--embedding-key', '-e',
                    default='small_bert/bert_en_uncased_L-4_H-128_A-2')
parser.add_argument('--run-id', type=str,
                    default=f'cluster_output-{str(uuid())}')
parser.add_argument('--algorithm', '-a', type=str, default='hdbscan', choices=[
    'hdbscan', 'dbscan', 'optics', 'kmeans', 'bisecting-kmeans'
])
parser.add_argument('--reduce-dim', '-r', action='store_true',
                    default=None, dest='reduce_dim')
parser.add_argument('--no-reduce-dim', action='store_false',
                    default=None, dest='reduce_dim')

parser.add_argument('--max-evals', '-m', type=int, default=100,
                    help='when running hp search, run this many trials')
parser.add_argument('--config', '-c', type=str, default='auto',
                    help='either "auto" in which case we do a parameter search, or the path to a json file with hyperparam values.')

logger = logging.getLogger('TEXT-CLUSTERING')
logger.setLevel(logging.INFO)


class TextEmbedder:

    def _create_vocabulary(self, dataset):
        return tf.data.Dataset.from_tensor_slices(
            set(k for k in dataset)
        )

    def _lookup_preproc(self, k):
        assert k in map_model_to_preprocess, f'Preprocessing model not found for {k}'
        return map_model_to_preprocess[k]

    def _lookup_encoder(self, k):
        assert k in map_name_to_handle, f'Encoder model not found for {k}'
        return map_name_to_handle[k]

    def build_embedder_fn(self):
        logger.info('Constructing embedder network')
        if self._embedding_key.startswith('sentence-transformers'):
            embed_fn = SentenceTransformer(self._embedding_key).encode
        else:
            preprocessor = hub.KerasLayer(
                self._lookup_preproc(self._embedding_key), name='preprocessor')
            encoder = hub.KerasLayer(self._lookup_encoder(self._embedding_key),
                                     name='encoder')

            def embed_fn(texts):
                return encoder(preprocessor(texts))['pooled_output']
        vector_length = embed_fn(['initialize']).shape[-1]
        return embed_fn, vector_length

    def __init__(self, embedding_key):
        self._embedding_key = embedding_key
        self._embed, self._vector_length = self.build_embedder_fn()

    def get_embedding_key(self):
        return self._embedding_key

    def embed(self, messages, batch_size=500):
        batches = []
        for i in range(0, len(messages), batch_size):
            batches.append(messages[i: min(
                i + batch_size, len(messages))])
        features = np.empty(shape=(0, self._vector_length))
        for batch in tqdm(batches, desc=f'Embedding raw text with {self._embedding_key}'):
            embedded_batch = self._embed(batch)
            # print(f'\n\nshape of embedded batch: {embedded_batch.shape}\n\n')
            features = np.concatenate([features, embedded_batch])
        return features


class TextClusterRun:

    def __init__(self, input_path: str, embedder: TextEmbedder, cluster_algo: str,
                 config='auto', reduce_dim=None, max_evals: int = 100):
        assert os.path.isfile(input_path), f'{input_path} does not exist'
        self._df = pd.read_csv(input_path)
        self._input_path = input_path
        self._embedder = embedder
        self._embedding_cacher = EmbeddingCacheManager(
            cache_dir='.cache/embeddings')
        self._config = config
        self._reduce_dim = reduce_dim
        self._algo_key = cluster_algo
        self._algo = self._get_algorithm_class()
        self._max_evals = max_evals
        if config == 'auto':
            self._init_param_space()
        else:
            self._init_static_config(config)

    def _init_static_config(self, config_path):
        with open(config_path) as cfile:
            dat = json.load(cfile)
        self._config_dat = dat

    def _get_algorithm_class(self):
        if self._algo_key == 'hdbscan':
            algo = hdbscan.HDBSCAN
        elif self._algo_key == 'optics':
            algo = OPTICS
        elif self._algo_key == 'affinity-prop':
            algo = AffinityPropagation
        elif self._algo_key == 'kmeans':
            algo = KMeans
        elif self._algo_key == 'bisecting-kmeans':
            algo = BisectingKMeans
        else:
            raise ValueError(
                f'{self._algo_key} is not a recognized cluster algorithm key')
        return algo

    def _init_param_space(self):
        self._param_space = {}
        if self._reduce_dim in (True, None):
            self._param_space['n_neighbors'] = hp.choice(
                'n_neighbors', range(5, 50))
            self._param_space['n_components'] = hp.choice(
                'n_components', range(5, 100))
            reduce_dim_conditions = [True]
            if self._reduce_dim is None:
                reduce_dim_conditions.append(False)
            self.reduce_dim = hp.choice('reduce_dim', reduce_dim_conditions)
        if self._algo_key == 'hdbscan':

            self._param_space['min_cluster_size'] = hp.choice(
                'min_cluster_size', range(2, 10))
            self._param_space['cluster_selection_epsilon'] = hp.uniform(
                'cluster_selection_epsilon', .1, 20)
            self._param_space['min_samples'] = hp.choice(
                'min_samples', range(6, 200))
            self._param_space['metric'] = hp.choice(
                'metric', ['euclidean', 'manhattan'])
            self._param_space['cluster_selection_method'] = hp.choice('selection', [
                                                                      'leaf', 'eom'])

        elif self._algo_key == 'optics':
            self._param_space['min_samples'] = hp.choice(
                'min_samples', range(6, 100))
            # self._param_space['max_eps'] = hp.uniform('max_eps', .5, 10)
            self._param_space['metric'] = hp.choice(
                'metric', ['manhattan', 'euclidean'])
            self._param_space['xi'] = hp.uniform('xi', 0, 1)
            self._param_space['leaf_size'] = hp.choice(
                'leaf_size', range(2, 10))
        elif self._algo_key == 'affinity-prop':
            self._param_space['damping'] = hp.uniform('damping', .5, 1)
        elif 'kmeans' in self._algo_key:
            self._param_space['n_clusters'] = hp.choice(
                'n_clusters', range(50, 1000))

    def _pct_noise(self, labels):
        n_noise = dict(zip(*np.unique(labels, return_counts=True))).get(-1, 0)
        return n_noise / len(labels)

    def _mean_sil_signal(self, embeddings, labels):
        e, l = [], []
        for i in range(len(labels)):
            if labels[i] != -1:
                e.append(embeddings[i])
                l.append(labels[i])
        return silhouette_score(e, l)

    def _score_clustering(self, clusters, embeddings, runner, threshold=0.05):
        # should be n - 1? does hdbscan include -1 label for noise?
        label_count = len(np.unique(clusters.labels_))
        total_num = len(clusters.labels_)
        penalty = 0
        if 'kmeans' in self._algo_key:
            cost = 1 - silhouette_score(embeddings, clusters.labels_)
        elif self._algo_key == 'hdbscan':
            # TODO: revisit this, but maybe not for a while, seems like HDBSCAN ain't it for this use case
            # cost = 1 - runner._relative_validity
            cost = (np.count_nonzero(clusters.probabilities_ <
                                     threshold) / total_num)
        elif self._algo_key == 'optics':
            # remove noise and get avg sil from those
            # print('scoring optics')
            n_clusters = len(set(clusters.labels_))
            # print(f'Cluster run has {n_clusters} clusters')
            if n_clusters <= 3:
                print('giving penalty score for too few clusters')
                sig_sil_loss = .3  # give VERY bad score for
                # such a low cluster count -- also silhouette_score isn't valid
                # for label_count == 1
            else:
                sig_sil_loss = 1 - \
                    self._mean_sil_signal(embeddings, clusters.labels_)
            noise_penalty = self._pct_noise(clusters.labels_) or 1
            size_variance = np.var(list({k: v for k, v in zip(
                *np.unique(clusters.labels_, return_counts=True)) if k != -1}.values())) or 1
            cost = sig_sil_loss * noise_penalty * size_variance
            cost /= label_count

        return label_count, cost

    def search_objective(self, params, embeddings):
        clusters, umap_embeddings, runner = self.generate_clusters(
            embeddings,
            params
        )
        n_clusters, cost = self._score_clustering(clusters, embeddings, runner)
        # TODO: assign penalties for undesired cluster counts?
        # Spectral, SNN-cliq, Seurat
        # OR use FAIL statuses in this case.
        # if n_clusters <= 50:
        #     cost += .05
        # return {'loss': cost, 'label_count': n_clusters, 'status': STATUS_OK if n_clusters > 1 else STATUS_FAIL}
        return {'loss': cost, 'label_count': n_clusters, 'status': STATUS_OK}

    def _get_std_algo_params(self):
        '''Get any params that were not searching over and want to remain static, but will vary
        as f(the algo were using)'''
        out = dict()
        if self._algo_key in ('hdbscan', 'optics'):
            out['memory'] = '.cache'
        if self._algo_key == 'hdbscan':
            out['gen_min_span_tree'] = True
        if self._algo_key in ('optics',):
            out['n_jobs'] = 1
        return out

    def generate_clusters(self, embeddings, algo_params):
        logger.debug('Running UMAP for dimensionality reduction')
        if 'kmeans' not in self._algo_key:  # doesn't support multiple dist metrics
            metric = algo_params.pop('metric')
            if metric == 'cosine':
                # hdbscan does not support cosine similarity
                # as a distance metric (see: https://github.com/scikit-learn-contrib/hdbscan/issues/69)
                # But euclidean distance on l2 normalized vectors is equivalent to cosine similarity
                x = normalize(x, norm='l2')
                use_metric = 'euclidean'
            else:
                use_metric = metric
            algo_params['metric'] = use_metric
        try:
            n_neighbors = algo_params.pop('n_neighbors')
            n_components = algo_params.pop('n_components')
        except KeyError:
            pass  # quick hack to support no-reduce-dim in cli opts
        if self._reduce_dim:
            # n_neighbors = algo_params.pop('n_neighbors')
            # n_components = algo_params.pop('n_components')
            x = umap.UMAP(
                n_neighbors=n_neighbors,
                n_components=n_components,
                metric='cosine'
            ).fit_transform(embeddings)
        else:
            x = embeddings
        logger.debug(f'Running {self._algo_key.upper()}')

        algo_params.update(self._get_std_algo_params())
        runner = self._algo(**algo_params)
        clusters = runner.fit(x)
        # runner.relative_validity_ <- get_algo_score should return this for HDBSCAN
        return clusters, x, runner

    def bayesian_search(self, embeddings, space, max_evals=100):
        trials = Trials()
        objective = partial(self.search_objective, embeddings=embeddings)
        logger.info(
            f'Finding optimal clustering configuration with {max_evals} maximum evaluations')
        best_trial = fmin(
            objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials
        )
        best_k = int(trials.best_trial['result']['label_count']) - 1
        best_params = space_eval(space, best_trial)
        logger.info(f'Best Parameters after search: {best_params}')
        logger.info(f'Optimized cluster count: {best_k}')
        best_clusters, umap_embeddings, best_score = self.generate_clusters(
            embeddings, best_params)
        n_clusters, cost = self._score_clustering(
            best_clusters, embeddings, best_score)
        return best_params, best_clusters, trials, umap_embeddings, cost

    def _create_tfidf_vectorizer(self):
        return TfidfVectorizer(use_idf=True, stop_words='english', ngram_range=(1, 3),
                               max_features=1000, min_df=0.05, token_pattern=r'(?u)\b[a-zA-Z0-9_.]{2,}\b',
                               analyzer='word'
                               )

    def _get_keywords(self, output) -> dict:
        init = self._create_tfidf_vectorizer()
        init.fit_transform(output['message'].tolist())
        vocab = init.vocabulary_
        keywords = {}
        for c in tqdm(sorted(set(output['label'].tolist())), desc='Summarizing Clusters'):
            tfidf = TfidfVectorizer(use_idf=True, stop_words='english', ngram_range=(1, 1),
                                    max_features=1000, min_df=0.05, token_pattern=r'(?u)\b[a-zA-Z0-9_.]{2,}\b',
                                    analyzer='word', vocabulary=vocab
                                    )
            cluster_entries = output[output['label'] == c]['message'].tolist()
            scores = tfidf.fit_transform(cluster_entries)
            top_5_idx = np.argsort([score.max()
                                   for score in scores.T.todense()])[-5:]
            keywords[c] = [tfidf.get_feature_names_out()[i] for i in top_5_idx]
        if -1 in keywords:
            del keywords[-1]
        return keywords

    def output_to_df(self, messages, message_ids, embeddings, clusters) -> Tuple[pd.DataFrame]:

        output = pd.DataFrame()

        labels = clusters.labels_
        output['message'] = messages
        output['label'] = labels
        generate_msg_id = (message_ids is None)
        # output['embedding'] = embeddings

        sil_scores = silhouette_samples(
            embeddings, labels)
        output['silhouette_score'] = sil_scores
        if generate_msg_id:
            output['message_id'] = [str(uuid()) for i in range(len(messages))]
        else:
            output['message_id'] = message_ids

        cluster_summary = pd.DataFrame()
        cluster_summary['label'] = sorted(
            set(c for c in output['label'] if c >= 0))
        keywords = self._get_keywords(output)
        cluster_sizes = output.groupby('label').value_counts().to_dict()
        cluster_summary['cluster_id'] = [str(uuid())
                                         for i in range(len(cluster_summary))]
        sil_map = output.groupby(
            'label')['silhouette_score'].apply(lambda x: x.mean())
        cluster_summary['silhouette_score'] = cluster_summary['cluster_id'].map(
            sil_map).round(5)
        if self._algo_key == 'hdbscan':
            cluster_summary['persistence'] = clusters.cluster_persistence_
        cluster_summary['keywords'] = cluster_summary['label'].map(keywords)
        cluster_summary['size'] = cluster_summary['label'].map(cluster_sizes)
        cluster_summary.sort_values('size', inplace=True, ascending=False)
        cluster_summary.to_csv('summary_dump.csv')
        output_summary = pd.DataFrame()
        total_messages = len(output)
        output_summary['total_messages'] = [total_messages]
        n_noise = len(output[output['label'] == -1])
        output_summary['messages_clustered'] = total_messages - n_noise
        output_summary['noise_message_count'] = n_noise
        output_summary['pct_noise'] = round(n_noise / total_messages, 5) * 100
        output_summary['avg_silhouette'] = output['silhouette_score'].mean()

        return output, cluster_summary, output_summary

    def run(self, output, run_id, message_id_tag='message_id'):

        output_root = output or os.getcwd()
        if os.path.isdir(os.path.join(output_root, 'data', run_id)):
            raise FileExistsError(os.path.isdir(
                os.path.join(output_root, 'data', run_id)))
        # self._df = self._data_loader.get_dataframe()
        messages = self._df['message'].tolist()
        if message_id_tag in self._df:
            # preserve original message id column if present
            message_ids = self._df[message_id_tag]
        else:
            message_ids = None
        trials = None

        cached_embeddings = self._embedding_cacher.check_cache(
            self._input_path, embedder_url=self._embedder._embedding_key)
        if cached_embeddings is None:
            logger.info('Generating document embeddings')
            embeddings = self._embedder.embed(messages)
        else:
            logger.info(
                f'using cached embeddings from {self._embedder._embedding_key}')
            embeddings = cached_embeddings

        self._embedding_cacher.cache_embeddings(self._input_path, embeddings,
                                                embedder_url=self._embedder._embedding_key)

        if self._config == 'auto':  # use bayesian search to find optimal configuration
            logger.info('Using bayesian search to find optimal configuration')
            # space = {
            #     'n_neighbors': self.n_neighbors,
            #     'n_components': self.n_components,
            #     'min_cluster_size': self.min_cluster_size,
            #     'sel_eps': self.cluster_selection_epsilon,
            #     'min_samples': self.min_samples,
            #     'metric': self.metric,
            #     'selection': self.selection,
            #     'reduce_dim': self.reduce_dim
            #     #'epsilon': self._eps
            #     # 'random_state': self.random_state
            # }
            best_params, best_clusters, trials, reduced_embeddings, best_score = self.bayesian_search(
                embeddings, self._param_space, max_evals=self._max_evals)

        # use a single config, saves time if you've already established an optimal one
        # elif isinstance(self._config, dict):
        #     logger.info(f'Using fixed configuration: {self._config}')
        #     best_params, best_clusters, reduced_embeddings = self.generate_clusters(
        #         embeddings, **self._config_dat)
        else:
            best_clusters, embs, runner = self.generate_clusters(
                embeddings, self._config_dat)
            # raise ValueError(
            #     f'Unrecognized config {self._config} of type: {type(self._config)}')

        str_complete = datetime.now().isoformat()

        logger.info(f'Saving output to {os.path.join(output_root, run_id)}')
        out_path = os.path.join(output_root, 'data', 'output', run_id)
        os.mkdir(out_path)
        output, cluster_summary, output_summary = self.output_to_df(
            messages, message_ids, reduced_embeddings, best_clusters)

        best_params['embedder'] = self._embedder.get_embedding_key()
        best_params['input_file'] = self._input_path
        best_params['time_finished'] = str_complete
        best_params['validity_score'] = best_score
        for fname, dataframe in [
            ('cluster_summary.csv', cluster_summary),
                ('run_summary.csv', output_summary)]:
            dataframe.to_csv(os.path.join(out_path, fname), index=False)
        output.reset_index()
        output.sort_values(by='label', ascending=False)
        print('OUTPUT DEBUG')
        print(output)
        output.to_csv(os.path.join(out_path, 'samples.csv'), index=False)
        print(f'BEST PARAMS: {best_params}')
        with open(os.path.join(out_path, 'run_params.json'), 'w') as dfile:
            dfile.write(json.dumps(best_params, indent=4))
        if trials is not None:

            pk_name = f'trials/last-{self._algo_key}-run.pkl'
            with open(pk_name, 'wb') as trial_dump:
                pk.dump(trials, trial_dump)


if __name__ == '__main__':
    output_path = ''
    args = parser.parse_args()
    # input_path = sys.argv[1].strip()
    logger.info(f'input: {args.input_path}')
    logger.info(f'output: {args.output}')
    logger.info(f'run_id: {args.run_id}')
    logger.info(f'using algorithm: {args.algorithm}')
    logger.info(f'using tensorflow embedding model: {args.embedding_key}')
    print()
    embedder = TextEmbedder(args.embedding_key)

    clustering = TextClusterRun(
        args.input_path, embedder, args.algorithm,
        config=args.config, reduce_dim=args.reduce_dim, max_evals=args.max_evals)

    clustering.run(args.output, args.run_id)
