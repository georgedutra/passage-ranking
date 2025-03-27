from load_dataset import DatasetLoader
from model import MSDRanker
import pandas as pd
from sklearn.metrics import log_loss
from scipy.special import softmax
from tqdm import tqdm

def rank_dataset(batch_size = 1000):
    dataset = DatasetLoader("msmarco-passage-v2/train")
    q_train, q_test = dataset.split_test_train(train_size=0.8)
    model = MSDRanker([])

    for row in q_train.itertuples(False):
        query = row.query
        query_id = row.query_id
        true_doc_id = row.doc_id
        
        docs_list = []
        scores_list = []
    
        for i in range(0, dataset.docs_count, batch_size):
            minibatch = dataset.full_load_docs(i, i+batch_size)

            model.recalc_bm25(minibatch["doc"].tolist())
            docs, scores = model.bm25_filter(query)
            
            docs_list.extend(docs)
            scores_list.extend(scores)

        # achar os maiores scores
        scores_df = pd.DataFrame({"doc_id": docs_list, "score": scores_list})
        sorted_scores = scores_df.sort_values(by="score", ascending=False)
        
        # Carrega os documentos de maior score
        corpus = dataset.load_docs_by_id(sorted_scores.head(10000)["doc_id"].tolist())
        results, scores = model.monobert_filter(corpus['doc'].tolist(), query, top_k=100)

        #re-rank
        corpus2 = [corpus['doc'].tolist()[i] for i in results]
        results, scores = model.duobert_filter(corpus2, query)
        
        true_doc_text = corpus[corpus["doc_id"] == true_doc_id]["doc"].values[0] # texto do documento relevante
        true_y = [0] * 5
        for i in range(5):
            if corpus2[results[i]] == true_doc_text:
                true_y[i] = 1
            else:
                true_y[i] = 0

        loss = log_loss(true_y, softmax(scores))
        print(f"Loss: {loss}")

if __name__ == "__main__":
    rank_dataset(1_000_000)