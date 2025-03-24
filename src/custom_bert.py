import torch
from transformers import BertModel, BertTokenizer

class BertSentenceEncoder(torch.nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert.to(self.device)
        
        self.digester = torch.nn.Sequential(
            torch.nn.Linear(768, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128)
        ).to(self.device)

    def forward(self, sentences):
        # Tokenize input sentences
        encoded_input = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        ).to(self.device)
        
        # Get BERT outputs
        with torch.no_grad():
            outputs = self.bert(**encoded_input)
        
        # Perform mean pooling with attention mask
        token_embeddings = outputs.last_hidden_state
        attention_mask = encoded_input['attention_mask']
        
        # Expand attention mask to match embedding dimensions
        input_mask_expanded = (
            attention_mask
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )
        
        # Sum embeddings along sequence dimension
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        
        # Calculate sum of mask values (number of real tokens)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Compute final embeddings
        sentence_embeddings = sum_embeddings / sum_mask
        
        # Pass embeddings through digester
        # This is the trainable part of the model
        sentence_embeddings = self.digester(sentence_embeddings)
        
        return sentence_embeddings
    
    def get_scores(self, corpus, query):
        # Get embeddings
        embeddings = self(corpus)
        query_embedding = self(query)
        
        # Compute scores
        scores = embeddings @ query_embedding.T
        
        return scores

# Usage example
if __name__ == "__main__":
    # Initialize model
    encoder = BertSentenceEncoder()
    
    # Example sentences
    sentences = [
        "a cat is a feline and likes to purr",
        "a dog is the human's best friend and loves to play",
        "a bird is a beautiful animal that can fly",
        "a fish is a creature that lives in water and swims",
    ]
    
    query = "does the fish purr like a cat?"
    
    # Get scores
    scores = encoder.get_scores(sentences, query)
    
    print(scores)