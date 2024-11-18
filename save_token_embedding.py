import torch
from transformers import BertModel

# Define the path to the BERT model
model_path = "bert-base-uncased"
# model_path = "bert-large-uncased"

def main():
    # Load the pre-trained BERT model
    bert_model = BertModel.from_pretrained(model_path)
    
    # Retrieve the input embeddings from the BERT model
    embeddings = bert_model.get_input_embeddings().weight
    
    # Save the embeddings to a file
    with open("tokens/textual.pth", "wb") as file:
        torch.save(embeddings, file)
    
    # Print the shape of the embeddings
    print(f"Embedding shape: {embeddings.shape}")

if __name__ == "__main__":
    main()