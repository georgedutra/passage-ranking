import ir_datasets
import get_docs as gd
import old.classify as cl
import pandas as pd


def get_number_samples(n) -> dict:
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

def create_sub_dataset(n: int, docs_per_query: int = 10):
    dataset = ir_datasets.load("msmarco-passage-v2/train")
    number_samples = get_number_samples(n)
    query_df = cl.get_query_df(dataset)
    sampled_queries = []
    
    for key, value in number_samples.items():
        try:
            sampled_subset = query_df[query_df["Class"] == key].sample(n=value, random_state=42)
            sampled_queries.append(sampled_subset)
        except:
            print("Não existem questões do tipo: ", key)

    query_df_filtered = pd.concat(sampled_queries, ignore_index=True)  # Juntar os dados amostrados
    
    ids_docs = gd.get_docs_split(dataset,
                                 list(query_df_filtered["Query_ID"]),
                                 docs_per_query)
    
    return query_df_filtered, ids_docs

if __name__ == "__main__":
    querys, ids_docs = create_sub_dataset(1000)
    print(len(querys))
    print(len(ids_docs))
