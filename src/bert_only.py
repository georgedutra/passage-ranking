import numpy as np
import pandas as pd

import torch

from custom_bert import BertSentenceEncoder

import os
import time
import pickle
import random
random.seed(42)

def load_dataset(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

dataset = "subset_msmarco_train_0.01_99.pkl"

# Load dataset
data = load_dataset(dataset)

# shuffle and sample queries to train/test/split
queries = data['queries']
query_ids = list(queries.keys())

random.shuffle(query_ids)

split_ratio = 0.8

train_query_ids = query_ids[:int(len(query_ids) * split_ratio)]
test_query_ids = query_ids[int(len(query_ids) * split_ratio):]

train_queries = {qid: queries[qid] for qid in train_query_ids}
test_queries = {qid: queries[qid] for qid in test_query_ids}

###

# make qrels easier to manipulate
qrels = []

for qrel in data['qrels']:
    query_id = qrel[0]
    doc_id = qrel[1]
    
    qrels.append((query_id, doc_id))

qrels = pd.DataFrame(qrels, columns=["query_id", "doc_id"])

# making easier to handle docs and queries
query_ids = np.array(query_ids)
doc_ids = np.array(list(data['docs'].keys()))

# create new positional ids for docs and create a dict to map it to old ids
new_doc_ids = {}

inx = 0
for doc_id in doc_ids:
    new_doc_ids[inx] = doc_id
    inx += 1

train_query_ids = np.array(train_query_ids)
test_query_ids = np.array(test_query_ids)

# create model
model = BertSentenceEncoder()

# function to create docs embeddings matrix
def predigest_docs(doc_ids, new_doc_ids, model):
    # reserve space in memory
    embeddings = torch.zeros([len(doc_ids), 768])

    # calculate and store embeddings for each doc
    for i in range(len(doc_ids)):
        embeddings[i, :] = model.get_embeddings(data['docs'][new_doc_ids[i]][1])
        
        # just to keep track of progress
        if i % 1000 == 0:
            print(i)

    # save matrix
    with open('predigested_docs.pkl', 'wb') as f:
        pickle.dump(embeddings, f)

# if matrix wasn't already calculated, calulate it
if not os.path.exists('predigested_docs.pkl'):
    predigest_docs(doc_ids, new_doc_ids, model)

# load matrix
with open('predigested_docs.pkl', 'rb') as f:
    docs_embeddings = pickle.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
docs_embeddings = docs_embeddings.to(device)
model.to(device)

# this function suggest k docs to answer a given query
def evaluate_query(model, query, num_docs, docs_embeddings):
    query_embedding = model.get_embeddings(query).to(device)
    
    scores = docs_embeddings @ query_embedding.T
    scores = scores[:, 0]
    
    return torch.argsort(scores, descending=True)[:num_docs]

# gets back docs texts from positional indices
def retrieve_docs(doc_ids, new_doc_ids, docs):
    result = []
    
    for doc_inx in doc_ids:
        old_doc_id = new_doc_ids[int(doc_inx)]
        
        result.append(docs[old_doc_id][1])
    
    return result

# Sanity check
# query_id = query_ids[100]
# query = data['queries'][query_id][1]
# print(query)

# doc_inx = evaluate_query(model, query, 10, docs_embeddings)
# print(doc_inx)

# docs = retrieve_docs(doc_inx, new_doc_ids, data['docs'])
# print(docs)

# print(qrels[qrels['query_id'] == query_id])
# print([new_doc_ids[int(inx)] for inx in doc_inx])

def time_check(model, queries, num_docs, docs_embeddings, new_doc_ids, docs):
    times = []
    for query_id, query in queries.items():
        query = query[1]
        
        start = time.time()
        doc_inx = evaluate_query(model, query, num_docs, docs_embeddings)
        doc = retrieve_docs(doc_inx, new_doc_ids, docs)
        end = time.time()
        
        times.append(end - start)
    
    times = np.array(times)
    return times.mean()

#print("Tempo médio para recuperar um documento para uma query:", time_check(model, test_queries, 1, docs_embeddings, new_doc_ids, data['docs']))

def get_precision(model, queries, num_docs, docs_embeddings, new_doc_ids, qrels):
    correct = 0
    for query_id, query in queries.items():
        query = query[1]
        
        doc_inx = evaluate_query(model, query, num_docs, docs_embeddings)
        if num_docs == 1:
            old_doc_id = new_doc_ids[int(doc_inx)]
            
            if old_doc_id in list(qrels[qrels['query_id'] == query_id]['doc_id']):
                correct += 1
        else:
            for doc_i in doc_inx:
                old_doc_id = new_doc_ids[int(doc_i)]
                
                if old_doc_id in list(qrels[qrels['query_id'] == query_id]['doc_id']):
                    correct += 1
        
    return correct / len(queries.keys()) * 100


# print("Precisão do modelo no teste:", get_precision(model, queries, 1, docs_embeddings, new_doc_ids, qrels))
# print("Precisão@5 do modelo no teste:", get_precision(model, queries, 5, docs_embeddings, new_doc_ids, qrels))
# print("Precisão@10 do modelo no teste:", get_precision(model, queries, 10, docs_embeddings, new_doc_ids, qrels))


def mrr(model, queries, num_docs, docs_embeddings, new_doc_ids, qrels):
    reciprocal_ranks = []
    for query_id, query in queries.items():
        query = query[1]
        
        doc_inx = evaluate_query(model, query, num_docs, docs_embeddings)
        for i in range(len(doc_inx)):
            doc_i = doc_inx[i]
            old_doc_id = new_doc_ids[int(doc_i)]
            
            if old_doc_id in list(qrels[qrels['query_id'] == query_id]['doc_id']):
                reciprocal_ranks.append(1 / (i + 1))
                continue
    
    reciprocal_ranks = np.array(reciprocal_ranks)
    num_queries = len(queries.keys())
    
    return reciprocal_ranks.sum() / num_queries

print("MRR@5:", mrr(model, queries, 5, docs_embeddings, new_doc_ids, qrels))
print("MRR@10:", mrr(model, queries, 10, docs_embeddings, new_doc_ids, qrels))