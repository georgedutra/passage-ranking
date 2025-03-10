import ir_datasets
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import ir_datasets
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

dataset_train = ir_datasets.load("msmarco-passage/train")

def sep(sep="=", time=42):
    print(sep*time)

sep()
print(f"Tipo de dataset: {dataset_train}")
print(f"Número de queries: {dataset_train.queries_count()}")
print(f"Queries cls fields: {dataset_train.queries_cls()._fields}")
print(f"Número de docs: {dataset_train.docs_count()}")
print(f"Queries cls fields: {dataset_train.queries_cls()._fields}")
sep()

def classify_query(text):
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


queries = [query.text for query in dataset_train.queries_iter()]
queries_id = [query.query_id for query in dataset_train.queries_iter()]

i = 0
queries_class = []
for query, query_id in zip(queries, queries_id):
    classe = classify_query(query)
    queries_class.append((query, query_id, classe))
    if i % 1000 == 0:    
        print(f"{i} de 808731")
    i += 1    

df = pd.DataFrame(queries_class, columns=["Query", "Query_ID", "Class"])
df.to_csv("queries_classified.csv", index=False)

class_counts = df["Class"].value_counts()
print(class_counts)

plt.figure(figsize=(10, 6))
class_counts.plot(kind="bar", color="skyblue", edgecolor="black")
plt.xlabel("Classes")
plt.ylabel("Quantidade")
plt.title("Distribuição das Classes de Queries")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)


plt.show()