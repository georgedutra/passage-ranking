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
        
        split = self.split_test_train()
        self.train_idx = split[0]
        self.test_idx = split[1]

    def load_queries(self) -> pd.DataFrame:
        """
        Loads queries from the given dataset and returns them as a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame with two columns:
                - "query_id": The unique identifier for each query.
                - "query": The text of the query.
        """
        dict = {"query_id": [], "query": []}

        a = [0, 2, 4]
        for data in self.dataset.queries_iter()[a]:
            dict["query_id"].append(data.query_id)
            dict["query"].append(data.text)

        df = pd.DataFrame(dict)
        return df
    
    def lazy_load_docs(self, num_docs: int = -1):
        """
        Lazily loads documents from the given dataset.
        Args:
            num_docs (int, optional): The number of documents to load. If set to -1, all documents 
                in the dataset will be loaded. Defaults to -1.
        Yields:
            tuple: A tuple containing the document ID (str) and the document text (str).
        """
        if num_docs == -1:
            num_docs = self.dataset.docs_count()

        for doc in self.dataset.docs_iter()[:num_docs]:
            doc_text = doc.text
            doc_id = doc.doc_id
            yield doc_id, doc_text

    def full_load_docs(self, num_docs: int = -1) -> pd.DataFrame:
        """
        Loads documents from the given dataset into a pandas DataFrame.
        Probably won't work witk msmarco dataset, since it has more than 20Gb of data.

        Args:
            num_docs (int, optional): The maximum number of documents to load. 
                If set to -1, all documents will be loaded. Defaults to -1.
        Returns:
            pd.DataFrame: A DataFrame containing the loaded documents with two columns:
                - 'doc_id': The unique identifier for each document.
                - 'doc_text': The text content of each document.
        """
        dict = {"doc_id": [], "doc": []}
        
        for doc_id, doc_text in self.lazy_load_docs(self.dataset, num_docs):
            dict["doc_id"].append(doc_id)
            dict["doc"].append(doc_text)

        df = pd.DataFrame(dict)
        return df

    def split_test_train(self, train_size: float = 0.8) -> tuple:
        """
        Splits the dataset into training and testing indices.
        Args:
            train_size (float, optional): Proportion of the dataset to include in the training split. 
                                          Must be between 0.0 and 1.0. Defaults to 0.8.
        Returns:
            tuple: A tuple containing two lists:
                - train_idx (list): Indices for the training set.
                - test_idx (list): Indices for the testing set.
        """
        num_docs = self.dataset.docs_count()
        train_idx, test_idx = train_test_split(range(num_docs), train_size=train_size)
        
        return train_idx, test_idx

    def load_train_docs(self):
        """
        Generator function to load training documents based on specified indices.
        Yields:
            tuple: A tuple containing:
                - doc_id (str): The unique identifier of the document.
                - doc_text (str): The text content of the document.
        Notes:
            - This function iterates over all documents in the dataset but only
              yields documents whose indices are present in `self.train_idx`
        """
        i = 0
        for doc in self.dataset.docs_iter():
            if i not in self.train_idx:
                continue

            doc_text = doc.text
            doc_id = doc.doc_id
            yield doc_id, doc_text
    
    def load_test_docs(self):
        """
        Generator function to load test documents from the dataset.
        Yields:
            tuple: A tuple containing the document ID (`doc_id`) and the document text (`doc_text`)
            for each document in the dataset that is part of the test set.
        Notes:
            - The method iterates over all documents in the dataset using `docs_iter()`.
            - Only documents whose indices are present in `self.test_idx` are yielded.
        """
        i = 0
        for doc in self.dataset.docs_iter():
            if i not in self.test_idx:
                continue

            doc_text = doc.text
            doc_id = doc.doc_id
            yield doc_id, doc_text
        

if __name__ == "__main__":
    # Might take a while to load for the first time
    loader = DatasetLoader("msmarco-passage-v2/train")
    queries = loader.load_queries()
    
    from IPython.display import display
    display(queries) 
    
    docs = loader.full_load_docs(10)
    display(docs) 
    