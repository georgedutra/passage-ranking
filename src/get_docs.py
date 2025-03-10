import heapq

def get_docs(dataset, query_id: str, num_docs: int = 10) -> tuple[list[str], list[str]]:
    """Finds the most and least relevant documents for a given query.

    Args:
        dataset: msmarco dataset
        query_id (str): query to evaluate
        num_docs (int): number of most and least relevant documents to return (default 10)

    Returns:
        Tuple[list[str], list[str]]: most relevant documents, least relevant documents
    """
    most_relevant_heap = []
    least_relevant_heap = []
    
    for qrel in dataset.scoreddocs_iter():
        if qrel['query_id'] == query_id:
            score = qrel['score']
            doc_id = qrel['doc_id']
            
            if len(most_relevant_heap) < num_docs:
                heapq.heappush(most_relevant_heap, (score, doc_id))
            else:
                if score > most_relevant_heap[0][0]:
                    heapq.heappop(most_relevant_heap)
                    heapq.heappush(most_relevant_heap, (score, doc_id))
            
            if len(least_relevant_heap) < num_docs:
                heapq.heappush(least_relevant_heap, (-score, doc_id))
            else:
                if -score > least_relevant_heap[0][0]:
                    heapq.heappop(least_relevant_heap)
                    heapq.heappush(least_relevant_heap, (-score, doc_id))
    
    # Extract the document IDs from the heap
    most_relevant_docs = [doc_id for (score, doc_id) in most_relevant_heap]
    least_relevant_docs = [doc_id for (score, doc_id) in least_relevant_heap]
    
    return most_relevant_docs, least_relevant_docs

def get_docs_split(dataset, queries_list: list[str], num_docs_per_query: int = 10) -> Set[str]:
    """Returns a set of filtered document IDs for a list of queries.

    Args:
        dataset: msmarco dataset
        queries_list (list[str]): query IDs to evaluate
        num_docs_per_query (int): number of most and least relevant documents to return

    Returns:
        Set[str]: set of document IDs
    """
    doc_ids = set()
    for query_id in queries_list:
        most_relevant_docs, least_relevant_docs = get_docs(dataset, query_id, num_docs_per_query)
        doc_ids.update(most_relevant_docs)
        doc_ids.update(least_relevant_docs)
    
    return doc_ids