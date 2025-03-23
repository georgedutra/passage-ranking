import bm25s

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
        self.retriever = bm25s.BM25(corpus=corpus)
        self.retriever.index(bm25s.tokenize(corpus))
        
        self.monobert = None
        self.duobert = None
    
    def bm25_filter(self, query: str, k: int = 5) -> tuple[list[str], list[float]]:
        """Applies BM25 filtering to a given query and corpus.

        Args:
            query (str): Query for which to retrieve relevant documents.
            k (int, optional): Number of top-k documents to retrieve. Defaults to 5.

        Returns:
            docs, scores (tuple[list[str], list[float]]): Tuple containing the list of top-k documents and their corresponding BM25 scores.
        """
        results, scores = self.retriever.retrieve(bm25s.tokenize(query), k=k)
        return results, scores

if __name__ == "__main__":
    corpus = [
        "a cat is a feline and likes to purr",
        "a dog is the human's best friend and loves to play",
        "a bird is a beautiful animal that can fly",
        "a fish is a creature that lives in water and swims",
    ]
    
    query = "does the fish purr like a cat?"
    
    ranker = MSDRanker(corpus)
    docs, scores = ranker.bm25_filter(query, k=2)
    print(docs, scores)