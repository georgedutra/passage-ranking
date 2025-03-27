import ir_datasets
import pandas as pd
from sklearn.model_selection import train_test_split

class DatasetLoader:
    """
    A class to load and manage datasets for information retrieval tasks.
    """
    def __init__(self, dataset_name: str):
        """
        Initializes the DatasetLoader with the given dataset name.
        Args:
            dataset_name (str): The name of the dataset to load.
        """
        self.dataset: ir_datasets.Dataset = ir_datasets.load(dataset_name)
        self.docs_count = self.dataset.docs_count()

    def load_queries(self) -> pd.DataFrame:
        """
        Loads queries from the given dataset and returns them as a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame with two columns:
                - "query_id": The unique identifier for each query.
                - "query": The text of the query.
                - "doc_id": The document ID associated with the query.
        """
        dict = {"query_id": [], "query": []}

        for data in self.dataset.queries_iter():
            dict["query_id"].append(data.query_id)
            dict["query"].append(data.text)

        df = pd.DataFrame(dict)

        dict_2 = {"query_id": [], "doc_id": []}
        for qrel in self.dataset.qrels_iter():
            dict_2["query_id"].append(qrel.query_id)
            dict_2["doc_id"].append(qrel.doc_id)
        
        df_2 = pd.DataFrame(dict_2)
        df = pd.merge(df, df_2, on="query_id")

        return df
    
    def split_test_train(self, train_size: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the dataset into training and testing sets based on the specified train size.
        Args:
            train_size (float): Proportion of the dataset to include in the training set. 
                                Must be a float between 0.0 and 1.0. Default is 0.8.
        Returns:
            tuple: A tuple containing two DataFrames:
                - The first DataFrame contains the training queries.
                - The second DataFrame contains the testing queries.
        """
        num_queries = self.dataset.queries_count()
        train_idx, test_idx = train_test_split(range(num_queries), train_size=train_size)
        
        queries = self.load_queries()
        train_queries = queries.iloc[train_idx]
        test_queries = queries.iloc[test_idx]

        return train_queries, test_queries
    
    def lazy_load_docs(self, idx_start: int = 0, idx_end: int = 0):
        """
        Lazily loads documents from the given dataset.
        Args:
            idx_start (int, optional): The starting index of the documents to load (inclusive). Defaults to 0.
            idx_end (int, optional): The ending index of the documents to load (exclusive). 
                If set to 0, all documents will be loaded. Defaults to 0.
        Yields:
            tuple: A tuple containing:
                - str: The document ID.
                - str: The document text.
        """
        if idx_end == 0:
            idx_end = self.docs_count

        for doc in self.dataset.docs_iter()[idx_start : idx_end]:
            yield doc.doc_id, doc.text

    def full_load_docs(self, idx_start: int = 0, idx_end: int = -1) -> pd.DataFrame:
        """
        Loads documents from the given dataset into a pandas DataFrame.
        Probably won't work with the whole msmarco dataset, since it has more than 20Gb of data.

        Args:
            idx_start (int, optional): The starting index of the documents to load (inclusive). Defaults to 0.
            idx_end (int, optional): The ending index of the documents to load (exclusive). 
                If set to 0, all documents will be loaded. Defaults to 0.
        Returns:
            pd.DataFrame: A DataFrame containing the loaded documents with two columns:
                - 'doc_id': The unique identifier for each document.
                - 'doc_text': The text content of each document.
        """
        dict = {"doc_id": [], "doc": []}
        
        for doc_id, doc_text in self.lazy_load_docs(idx_start, idx_end):
            dict["doc_id"].append(doc_id)
            dict["doc"].append(doc_text)

        df = pd.DataFrame(dict)
        return df

    def load_docs_by_id(self, id_list: list):
        """
        Loads documents from the given dataset into a pandas DataFrame based on a list of document IDs.
        Args:
            id_list (list): A list of document IDs to load.
        Returns:
            pd.DataFrame: A DataFrame containing the loaded documents with two columns:
                - 'doc_id': The unique identifier for each document.
                - 'doc': The text content of each document.
        """
        dict = {"doc_id": [], "doc": []}
        
        for doc_id, doc_text in self.lazy_load_docs():
            if doc_id in id_list:
                dict["doc_id"].append(doc_id)
                dict["doc"].append(doc_text)

        df = pd.DataFrame(dict)
        return df
        

if __name__ == "__main__":
    # Might take a while to load for the first time
    loader = DatasetLoader("msmarco-passage-v2/train")
    queries = loader.load_queries()
    
    from IPython.display import display
    display(queries) 
    
    # docs = loader.full_load_docs(10)
    # display(docs) 
    