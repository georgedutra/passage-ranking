from nltk.tokenize import word_tokenize
from nltk import pos_tag
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
from ir_datasets.datasets.base import Dataset
import pandas as pd

def classify_query(text: str) -> str:
    """
    Classifies a query based on the presence of interrogative words (WH-words) 
    or auxiliary verbs, using POS tagging.

    Args:
        text (str): The query text to classify.

    Returns:
        str: The classified type of the query, one of:
            - "What"
            - "Which"
            - "How"
            - "Who"
            - "Where"
            - "When"
            - "Why"
            - "YesNo"
            - "Other"
    """
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)

    if not pos_tags:  
        return "Other"

    for word, tag in pos_tags:
        word_lower = word.lower()
        
        if tag in ["WP", "WP$", "WRB"]:  
            if word_lower in ["what"]:
                return "What"
            elif word_lower in ["which"]:
                return "Which"
            elif word_lower in ["how"]:
                return "How"
            elif word_lower in ["who", "whom", "whose"]:
                return "Who"
            elif word_lower in ["where"]:
                return "Where"
            elif word_lower in ["when"]:
                return "When"
            elif word_lower in ["why"]:
                return "Why"
    
    first_word = pos_tags[0][0].lower()
    if first_word in ["is", "are", "does", "do", "did", "can", "will", "should"]:
        return "YesNo"
    
    return "Other"

def get_query_df(dataset: Dataset) -> pd.DataFrame:
    """
    Generates a Pandas DataFrame containing queries, their IDs, and their classified types.

    Args:
        dataset (Dataset): An `ir_datasets` dataset object that provides query data.

    Returns:
        pd.DataFrame: A DataFrame with three columns:
            - "Query": The original query text.
            - "Query_ID": The unique identifier for the query.
            - "Class": The classified type of the query.
    """
    df = pd.DataFrame(columns=["Query", "Query_ID", "Class"])
    
    for data in dataset.queries_iter():
        query_type = classify_query(data.text)
        df.loc[len(df)] = [data.text, data.query_id, query_type]

    return df