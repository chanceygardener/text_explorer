import pandas as pd
import numpy as np
import re
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances
import tensorflow_hub as hub
from embedding_cache import EmbeddingCacheManager
from pretrained_url import map_model_to_preprocess, map_name_to_handle


class SemanticSimilaritySearch:

    class SemanticEncoder:
        def __init__(self, embedding_key: str,
                     space: pd.DataFrame, space_path: str,
                     embedding_cache: EmbeddingCacheManager):
            self._embedding_key = embedding_key
            if self._embedding_key.startswith('sentence-transformers'):
                embed_fn = SentenceTransformer(embedding_key).encode
            else:
                preprocessor = hub.KerasLayer(
                    self._lookup_preproc(self._embedding_key), name='preprocessor')
                encoder = hub.KerasLayer(self._lookup_encoder(self._embedding_key),
                                         name='encoder')

                def embed_fn(texts):
                    return encoder(preprocessor(texts))['pooled_output']
            self._embed = embed_fn
            self._space = space
            
            embeddings = embedding_cache.check_cache(
                space_path, embedder_url=self._embedding_key
            )
            if not embeddings:
                print(f'embedding search space with {self._embedding_key}')    
                embeddings = list(
                    self._embed(self._space.message.tolist()))
                embedding_cache.cache_embeddings(
                    space_path, embeddings, self._embedding_key)
            else:
                print(f'Using cached embeddings for {self._embedding_key}')
            self._space[self._embedding_key] = embeddings

        def _lookup_preproc(self, k):
            assert k in map_model_to_preprocess, f'Preprocessing model not found for {k}'
            return map_model_to_preprocess[k]

        def _lookup_encoder(self, k):
            assert k in map_name_to_handle, f'Encoder model not found for {k}'
            return map_name_to_handle[k]

        def _srange(self, df, msg=None, uid=None, min_sim=.6, embedding=None, sim_type='cosine'):
            assert msg or uid or embedding is not None, f'Must set one of: msg, uid'
            if msg:
                embedding = SentenceTransformer(key).encode([msg])[0]
                print(f'embedding from message is of shape: {embedding.shape}')
            elif uid:
                selction = df[df.message_id == uid]
                # print(
                #     f'Comparing to: {selction.iloc[0].message} embedding shape: {selction.iloc[0].embedding.shape}')
                # embedding = np.array([selction.iloc[0].embedding])
                embedding = selction.iloc[0][self._embedding_key]
            elif embedding is not None:
                pass
            df['similarity'] = df[self._embedding_key].apply(
                lambda e: 1 - cosine(e, embedding))
            return df[df.similarity >= min_sim]

        def _centroid(self, embeddings):
            return embeddings.mean(axis=0)

        def msg_search(self, text, min_sim=.9):
            return self._srange(self._space, msg=text, min_sim=min_sim)

        def id_search(self, msg_id, min_sim=.9):
            return self._srange(self._space, uid=msg_id, min_sim=min_sim)

        def group_search(self, messages, min_sim=.9):
            embedding = self._centroid(
                self._embed(messages)
            )
            return self._srange(self._space, embedding=embedding, min_sim=min_sim)

        def regex_text_match(self, pat):
            return self._space.filter(regex=pat)

        def text_search(self, text):
            return self._space[self._space.message.str.contains(text, na=False)]

    def get_executor(self, embedding_key):
        return self.SemanticEncoder(embedding_key, self._space,
                                    self._space_paths, self._cacher)

    def __init__(self, space_paths, column_mappers=None):
        self._space_paths = space_paths
        self._cacher = EmbeddingCacheManager('.cache/embeddings')
        self._space = self._init_space(space_paths, column_mappers)

    def _map_columns(self, space_paths, column_mappers):
        for p, m in zip(space_paths, column_mappers):
            df = pd.read_csv(p)
            for source, target in m.items():
                df[target] = df[source]
                del df[source]
            yield df

    def _init_space(self, space_paths, column_mappers):
        if column_mappers is not None:
            assert len(space_paths) == len(column_mappers), \
                'list of file paths must be the same as the length of the column mappers'
            assert type(space_paths) == type(column_mappers)
        if isinstance(space_paths, str):
            space_paths = [space_paths]
            if column_mappers is not None:
                column_mappers = [column_mappers]
        if column_mappers is not None:
            space_iter = self._map_columns(space_paths, column_mappers)
        else:
            space_iter = map(pd.read_csv, space_paths)
        return pd.concat([
            k for k in space_iter
        ])


def sem_sim(to, checkset, sim_type='cosine'):
    if sim_type == 'cosine':
        out = cosine_similarity(to, checkset)[0]
    elif sim_type == 'manhattan':
        out = manhattan_distances(to, checkset)[0]
        out = sigmoid(out)
        out = 1 - out
    else:
        raise ValueError(f'Unrecognized similarity type')
    return out


def sigmoid(z):
    return 1/(1 + np.exp(-z))


key = "sentence-transformers/sentence-t5-base"


def most_similar(df, msg=None, uid=None, n=10, sim_type='cosine'):
    assert msg or uid, f'Must set one of: msg, uid'
    look_at = df.copy()
    if msg:
        embedding = SentenceTransformer(key).encode([msg])
    elif uid:
        embedding = np.array([df[df.message_id == uid].iloc[0].embedding])
        look_at = look_at[look_at.message_id != uid]
    # similarities = sem_sim(embedding, look_at.embedding.to_numpy())
    similarities = sem_sim(embedding, np.array(
        look_at.embedding.tolist()), sim_type=sim_type)
    max_idxs = np.argpartition(similarities, -n)[-n:]
    return look_at.filter(items=max_idxs, axis=0)


def normalize(word_vec):
    norm = np.linalg.norm(word_vec)
    if norm == 0:
        return word_vec
    return word_vec/norm


def sem_range(df, msg=None, uid=None, min_sim=.6, sim_type='cosine'):
    assert msg or uid, f'Must set one of: msg, uid'
    if msg:
        embedding = SentenceTransformer(key).encode([msg])
    elif uid:
        selction = df[df.message_id == uid]
        print(f'Comparing to: {selction.iloc[0].message}')
        embedding = np.array([selction.iloc[0].embedding])
    similarities = sem_sim(embedding, np.array(
        df.embedding.tolist()), sim_type=sim_type)
    idxs = np.argwhere(similarities >= min_sim)
    # idxs = idxs.reshape(idxs.shape[0])
    return df.filter(items=idxs[0], axis=0)


def srange(df, msg=None, uid=None, min_sim=.6, embedding=None, sim_type='cosine'):
    assert msg or uid or embedding is not None, f'Must set one of: msg, uid'
    if msg:
        embedding = SentenceTransformer(key).encode([msg])
    elif uid:
        selction = df[df.message_id == uid]
        print(
            f'Comparing to: {selction.iloc[0].message} embedding shape: {selction.iloc[0].embedding.shape}')
        # embedding = np.array([selction.iloc[0].embedding])
        embedding = selction.iloc[0].embedding
    elif embedding is not None:
        pass
    df['similarity'] = df['embedding'].apply(
        lambda e: 1 - cosine(e, embedding))
    return df[df.similarity >= min_sim].sort_values(by=['similarity'], ascending=False)


def research(pat, df):
    return df[df.message.str.contains(pat, regex=True, na=False)]


def display(df):
    for i in range(len(df)):
        row = df.iloc[i]
        print(f'{row.message_id}: {row.message}\n')


def write(name, df):
    df.to_csv(name, index=False)


def lookup(uid, dat):
    return dat[dat.message_id == uid].iloc[0].message


def search(df, term):
    out = []
    for i in range(len(df)):
        row = df.iloc[i]
        if term.lower() in row.message:
            res = f'{row.message_id}: {row.message}'
            out.append(res)
            print(res, '\n')
    return out


def normalize(word_vec):
    norm = np.linalg.norm(word_vec)
    if norm == 0:
        return word_vec
    return word_vec/norm


# count = 0
# for i in range(len(fail)):
#     sim = 1 - cosine(normalize(t.embedding), normalize(fail.iloc[i].embedding))
#     if sim > thresh:
#         count += 1
#         print(
#             f'I think this is {sim} similar to:\n\t{fail.iloc[i].message_id}: {fail.iloc[i].message}')


# print(f'{count} sufficiently similar')


def intent_suggestion(intent_name, domain, source_dat, min_sim=.9):
    intent_data = source_dat[source_dat.intent_name == intent_name]
    centroid = SentenceTransformer(
        "sentence-transformers/sentence-t5-base").encode(
            intent_data.message.tolist()
    ).mean(axis=0)
    domain['similarity'] = domain['embedding'].apply(
        lambda e: 1 - cosine(e, centroid))
    return domain[domain.similarity >= min_sim]


# cosine(SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').encode(["i feel like i've failed."])[0], SentenceTransformer(
#     'sentence-transformers/all-MiniLM-L6-v2').encode(['i can smell the cherry blossoms, in the distance'])[0])

search_space = SemanticSimilaritySearch('unannotated.csv')
# search_space = SemanticSimilaritySearch('test.csv')


tfive_search = search_space.get_executor(key)
# id_check = tfive_search.id_search('34aa7d17-9fcb-425f-873f-2f7794ca7f55', min_sim=.6)
# regex = tfive_search.regex_text_match('.*offend.*')
# msg_sim = tfive_search.msg_search('I have codependency problems')
