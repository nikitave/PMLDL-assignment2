import pandas as pd
import implicit
from scipy.sparse import coo_matrix
import os


test_data = pd.read_csv('./benchmark/ua.test', sep='\t', header=None)
test_data = test_data.rename(columns={0: "user_id", 1: "item_id", 2: "rating", 3: "timestamp"})

all_items = test_data.item_id.unique()
test_data = test_data.drop('timestamp', axis=1)
test_data = test_data.groupby("user_id").agg(list)

def get_model():
    data = pd.read_csv('./data/raw/ua.base', sep='\t', header=None)
    data = data.rename(columns={0: "user_id", 1: "item_id", 2: "rating", 3: "timestamp"})
    sparse_matrix = coo_matrix((data['rating'].astype(float),
                            (data['user_id'], data['item_id'])))
    model = implicit.als.AlternatingLeastSquares(factors=10, regularization=0.02, iterations=600)
    model.fit(sparse_matrix)
    return model, sparse_matrix


m_p_at_k = []

model, sparse_matrix = get_model()

user_id = 1
user_items = sparse_matrix.T.tocsr()
recommendations = model.recommend(user_id, user_items[user_id])

for user_id, (item_ids, ratings) in test_data.iterrows():

    items = list(zip(item_ids, ratings))
    positive_items = list(map(lambda x: x[0], list(filter(lambda x: x[1] >= 3, items))))

    rec_mov_ids, _ = model.recommend(user_id, user_items[user_id], N=7)
    p_at_k = sum([1 if x in positive_items else 0 for x in rec_mov_ids]) / len(rec_mov_ids)
    m_p_at_k.append(p_at_k)

print("Precision is", sum(m_p_at_k) / len(m_p_at_k))