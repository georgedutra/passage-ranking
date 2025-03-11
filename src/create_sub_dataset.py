import ir_datasets
import get_docs as gd
import classify as cl
import pandas as pd
import json
import os
from logger import setup_logger

logger = setup_logger()

def get_number_samples(n) -> dict:
    """Computes the number of queries to sample for each category based on predefined percentage distribution.

    Args:
        n (int): Total number of queries to be sampled.

    Returns:
        dict[str, int]: A dictionary where keys are question categories and values are the number of queries to sample.
    """
    query_percentage = {
        "YesNo": 7.46,
        "What": 34.96,
        "How": 16.8,
        "Where": 3.46,
        "When": 2.71,
        "Why": 1.67,
        "Who": 3.33,
        "Which": 1.79,
        "Other": 27.83
    }
    number_samples = {}
    for key, value in query_percentage.items():
        number_samples[key] = max(1, int(n * value // 100))

    number_samples["Other"] = n - sum(number_samples.values())

    return number_samples

def create_sub_dataset( dataset, n: int, docs_per_query: int = 10)-> tuple[pd.DataFrame, set[str]]:
    """
    Creates a balanced subset of queries and documents from the MS MARCO Passage V2 dataset.

    Args:
        dataset (ir_datasets.datasets.base.Dataset): The MS MARCO dataset object.
        n (int): Total number of queries to be selected.
        docs_per_query (int, optional): Number of most and least relevant documents per query. Defaults to 10.

    Returns:
        tuple[pd.DataFrame, set[str]]: 
            - A DataFrame containing the sampled queries.
            - A set of document IDs for the most and least relevant documents.
    """    
    number_samples = get_number_samples(n)
    query_df = cl.get_query_df(dataset)
    sampled_queries = []

    for key, value in number_samples.items():
        try:
            sampled_subset = query_df[query_df["Class"] == key].sample(n=value, random_state=42)
            sampled_queries.append(sampled_subset)
            
        except:
            logger.warning(f"No queries of type '{key}' found in the dataset.")

    query_df_filtered = pd.concat(sampled_queries, ignore_index=True)  
    
    ids_docs = gd.get_docs_split(dataset,
                                 list(query_df_filtered["Query_ID"]),
                                 docs_per_query)
    
    return query_df_filtered, ids_docs

def related_ids_docs(dataset, ids_docs: set[str]) -> tuple[dict[str, str], dict[str, str]]: 
    """Get documents matching a list of IDs and create mapping dictionaries.

    Args:
        dataset (ir_datasets.datasets.base.Dataset): The MS MARCO dataset object.
        ids_docs (set[str]): Set of document IDs to retrieve.

    Returns:
        Tuple[dict[str, str], dict[str, str]]: 
            - `related_id_doc`: Dictionary that maps document IDs to their texts.
            - `related_doc_id`: Dictionary that maps document texts to their respective IDs.
    """
    docstore = dataset.docs_store()
    related_id_doc = {}
    related_doc_id = {}
    num_control = 0
    len_docs = len(ids_docs)

    for doc_id in ids_docs:
        try:
            text_doc = docstore.get(doc_id)  
            related_id_doc[doc_id] = text_doc.text 
            related_doc_id[text_doc.text] = doc_id 
            
            num_control += 1
            if num_control % 100 == 0:
                logger.debug(f"Processing document {num_control} of {len_docs}")
        
        except KeyError:
            logger.error(f"Document with ID {doc_id} not found.")
    
    return related_id_doc, related_doc_id
    
if __name__ == "__main__":
    logger.debug("Opening dataset...")
    dataset = ir_datasets.load("msmarco-passage-v2/train")
    
    logger.debug("Starting query and document selection process...")
    queries, ids_docs = create_sub_dataset(dataset, 1000)

    os.makedirs("data/csv", exist_ok=True)
    os.makedirs("data/json", exist_ok=True)
    
    logger.debug(f"Saving {len(queries)} queries to CSV...")
    queries.to_csv(os.path.join("data", "csv", "queries.csv"), index=False)

    logger.debug(f"Saving {len(ids_docs)} document IDs to CSV...")
    pd.DataFrame({"Document_IDs": list(ids_docs)}).to_csv(os.path.join("data", "csv", "ids_docs.csv"), index=False)


    logger.debug("Retrieving related documents...")
    related_id_doc, related_doc_id = related_ids_docs(dataset, ids_docs)  

    logger.debug(f"Saving {len(related_id_doc)} related_id_doc mappings to JSON...")
    with open(os.path.join("data", "json", "related_id_doc.json"), "w", encoding="utf-8") as f:
        json.dump(related_id_doc, f, ensure_ascii=False, indent=4) 

    logger.debug(f"Saving {len(related_doc_id)} related_doc_id mappings to JSON...")
    with open(os.path.join("data", "json", "related_doc_id.json"), "w", encoding="utf-8") as f:
        json.dump(related_doc_id, f, ensure_ascii=False, indent=4) 

    logger.debug("Processing completed successfully!")

    print(f"Total selected queries: {len(queries)}")
    print(f"Total retrieved documents: {len(ids_docs)}")