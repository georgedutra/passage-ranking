import bm25s
import numpy as np
import torch
from custom_bert import BertSentenceEncoder

class MSDRanker:
    """Multi-Stage Document Ranker.
    
    Stage 1: BM25 filtering.
    Stage 2: MonoBert scoring.
    Stage 3: DuoBert re-ranking.
    """
    def __init__(self, corpus: list[str]):
        """Initializes the MSDRanker with a given corpus.
        
        Args:
            corpus (list[str]): List of documents to index.
        """
        self.corpus = corpus
        self.retriever = bm25s.BM25(corpus=corpus)
        self.retriever.index(bm25s.tokenize(corpus))
        
        self.monobert = BertSentenceEncoder()
        self.duobert = BertSentenceEncoder(duobert=True)
    
    def bm25_filter(self, query: str, k: int = 5) -> tuple[np.ndarray[int], np.ndarray[float]]:
        """Applies BM25 filtering to a given query and corpus.

        Args:
            query (str): Query for which to retrieve relevant documents.
            k (int, optional): Number of top-k documents to retrieve. Defaults to 5.

        Returns:
            docs_inx, scores (np.array[int], np.array[float]): Tuple containing the list of top-k documents indices and their corresponding BM25 scores.
        """
        results, scores = self.retriever.retrieve(bm25s.tokenize(query), np.arange(len(self.corpus)), k=k)
        return results[0, :], scores[0, :]
        
    def monobert_filter(self, corpus, query: str, k: int = 5) -> tuple[np.ndarray[int], np.ndarray[float]]:
        """Applies MonoBert scoring to a given query and corpus.

        Args:
            query (str): Query for which to retrieve relevant documents.
            k (int, optional): Number of top-k documents to retrieve. Defaults to 5.

        Returns:
            docs_inx, scores (np.array[int], np.array[float]): Tuple containing the list of top-k documents indices and their corresponding MonoBert scores.
        """
        scores = self.monobert.get_scores(corpus, query)
        
        inx = torch.argsort(scores, dim=0, descending=True).squeeze().detach().cpu().numpy()
        
        results = inx[:k]
        
        scores = scores[results, 0]
        
        return results, scores
    
    def duobert_filter(self, corpus, query: str, k: int = 5) -> tuple[np.ndarray[int], np.ndarray[float]]:
        """Applies DuoBert re-ranking to a given query and corpus.

        Args:
            query (str): Query for which to retrieve relevant documents.
            k (int, optional): Number of top-k documents to retrieve. Defaults to 5.

        Returns:
            docs_inx, scores (np.array[int], np.array[float]): Tuple containing the list of top-k documents indices and their corresponding DuoBert scores.
        """
        scores = self.duobert.get_scores(corpus, query)
        
        inx = torch.argsort(scores, dim=0, descending=True).squeeze().detach().cpu().numpy()
        
        results = inx[:k]
        
        scores = scores[results]
        
        return results, scores
    
    def filter(self, query: str, k0: int = 3, k1: int = 2):
        docs, _ = self.bm25_filter(query, k=k0)
        
        filtered_corpus = [self.corpus[i] for i in docs]
        
        docs, _ = self.monobert_filter(filtered_corpus, query, k=k1)
        
        filtered_corpus = [filtered_corpus[i] for i in docs]
        
        docs, scores = self.duobert_filter(filtered_corpus, query, k=k1)
        
        return docs, scores
        

if __name__ == "__main__":
    corpus = [
        "a cat is a feline and likes to purr",
        "a dog is the human's best friend and loves to play",
        "a bird is a beautiful animal that can fly",
        "a fish is a creature that lives in water and swims",
    ]
    
    query = "does the fish purr like a cat?"
    
    print('===================')
    
    ranker = MSDRanker(corpus)
    docs, scores = ranker.bm25_filter(query, k=2)
    print(docs, scores)
    
    print('===================')
    
    results, scores = ranker.monobert_filter(corpus, query, k=2)
    print(results, scores)
    
    print('===================')
    
    results, scores = ranker.duobert_filter(corpus, query, k=2)
    print(results, scores)
    
    print('===================')
    print('===================')
    
    docs, scores = ranker.filter(query, k0=2, k1=2)
    
    print(docs, scores)
    print('===================')