import torch
import re
import logging
import numpy as np
import pandas as pd
from transformers import logging as hf_logging
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

hf_logging.set_verbosity_error()

def ngrams(string, n=4):

    """
    Generate n-grams from a given string. The process includes removing non-ascii characters, normalization, and cleaning.

    Args:
    string (str): The string to generate n-grams from.
    n (int): The number of characters in each n-gram.

    Returns:
    list: A list of n-grams generated from the string.
    """


    string = string.encode("ascii", errors="ignore").decode()
    string = string.lower()
    chars_to_remove = [")","(",".","|","[","]","{","}","'"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, '', string)
    string = string.replace('&', 'and')
    string = string.replace(',', ' ')
    string = string.replace('-', ' ')
    string = string.title()
    string = re.sub(' +',' ',string).strip()
    string = ' '+ string +' '
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

def prepare_LLM_data(full_training_set):
    """Prepare data for training: tokenize and create datasets."""
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Extract columns and convert to lists
    gbif_names = full_training_set['gbif_name'].astype(str).tolist()
    ncbi_names = full_training_set['ncbi_name'].astype(str).tolist()

    # Combine with separator token
    combined_texts = [gbif + " [SEP] " + ncbi for gbif, ncbi in zip(gbif_names, ncbi_names)]
    inputs = tokenizer(combined_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)

    # Labels for the dataset
    labels = torch.tensor(full_training_set['match'].tolist())

    # Split data into training and validation sets
    train_inputs, val_inputs, train_labels, val_labels = train_test_split(
        inputs['input_ids'], labels, test_size=0.1, random_state=42
    )
    return train_inputs, val_inputs, train_labels, val_labels

def create_dataloaders(train_inputs, val_inputs, train_labels, val_labels, batch_size=16):
    """Create DataLoader for training and validation."""
    # Convert inputs to TensorDataset
    train_data = TensorDataset(train_inputs, train_labels)
    val_data = TensorDataset(val_inputs, val_labels)

    # Samplers and DataLoader
    train_sampler = RandomSampler(train_data)
    val_sampler = SequentialSampler(val_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    return train_dataloader, val_dataloader

def initialize_model():
    """Initialize the BERT model for sequence classification."""
    return BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=2, output_attentions=False, output_hidden_states=False
    )

def train_model(model, train_dataloader, device, lr=2e-5, eps=1e-8, patience=5, max_epochs=100):
    """Train the model with early stopping."""
    optimizer = AdamW(model.parameters(), lr=lr, eps=eps)
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False

    model.train()
    for epoch_i in range(max_epochs):
        if early_stop:
            print("Early stopping!")
            break

        print(f'Epoch {epoch_i + 1}/{max_epochs}')
        running_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            b_input_ids, b_labels = batch[0].to(device), batch[1].to(device)
            model.zero_grad()
            outputs = model(b_input_ids, token_type_ids=None, labels=b_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_dataloader)
        print(f'Epoch {epoch_i + 1} Loss: {epoch_loss:.4f}')

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            early_stop = True
            print(f"No improvement in loss for {patience} consecutive epochs, stopping training.")

def evaluate_model(model, val_dataloader, device):
    """Evaluate the model, return accuracy."""
    model.eval()
    eval_accuracy = 0
    nb_eval_steps = 0

    for batch in val_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, labels=b_labels)

        logits = outputs.logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    return eval_accuracy / nb_eval_steps

def flat_accuracy(preds, labels):
    """Calculate accuracy of predictions."""
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
    
class TaxonomyDataset(Dataset):
    """Dataset for handling paired taxonomy strings."""
    def __init__(self, paired_strings):
        self.paired_strings = paired_strings
    
    def __len__(self):
        return len(self.paired_strings)
    
    def __getitem__(self, idx):
        return self.paired_strings[idx]
        
def train_and_evaluate_LLM(training_set, max_epochs):
    """
    Train and evaluate a machine learning model using predefined functions and settings.
    Returns the trained model, tokenizer, and validation accuracy.

    :param training_set: The dataset to be used for training and validation.
    :param max_epochs: Maximum number of epochs for training.
    :return: Tuple containing the trained model, tokenizer, and validation accuracy.
    """
    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Initialize the model 
    model = initialize_model()
    
    # Prepare the data 
    train_inputs, val_inputs, train_labels, val_labels = prepare_LLM_data(training_set)
    
    # Create the DataLoaders 
    train_dataloader, val_dataloader = create_dataloaders(train_inputs, val_inputs, train_labels, val_labels)
    
    # Define the device and move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Train the model 
    train_model(model, train_dataloader, device, max_epochs=max_epochs)
    
    # Evaluate the model 
    accuracy = evaluate_model(model, val_dataloader, device)
    
    # Print the validation accuracy
    print()
    print(f"Validation Accuracy: {accuracy:.4f}")

    # Return the trained model, tokenizer, and the accuracy for further use
    return model, tokenizer
        
def find_nearest_neighbors(query, target, n_neighbors=3, analyzer_func=ngrams):
    """
    Calculate nearest neighbors using TF-IDF and Nearest Neighbors.
    :param query: list of query strings
    :param target: list of target strings
    :param n_neighbors: Number of neighbors to find
    :param analyzer_func: Analysis function for the TF-IDF vectorizer
    :return: numpy array with queries, matching targets and distances
    """
    if analyzer_func is None:
        analyzer_func = 'word'  # Default analyzer
    
    # Configura il vettorizzatore TF-IDF
    vectorizer = TfidfVectorizer(analyzer=analyzer_func, lowercase=True)
    tfidf = vectorizer.fit_transform(target)
    
    # Configura il modello NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', n_jobs=-1, metric='cosine')
    nbrs.fit(tfidf)
    
    # Calcola le distanze e gli indici dei vicini
    distances, indices = nbrs.kneighbors(vectorizer.transform(query))
    distances = np.round(distances, 2)
    
    # Preparazione dei dati di output
    expanded_query = np.repeat(query, n_neighbors)
    expanded_target = np.array(target)[indices.flatten()]
    expanded_distances = distances.flatten()
    
    # Combinazione di query, target e distanze in un unico array
    matches = np.column_stack((expanded_query, expanded_target, expanded_distances))
    
    return matches

def create_data_loader(paired_strings, batch_size=16):
    """Create a DataLoader to manage batches for the dataset."""
    dataset = TaxonomyDataset(paired_strings)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def make_predictions(model, data_loader, tokenizer, device):
    """Run model predictions on the data loaded, outputting class probabilities."""
    model.eval()
    prob_class_positives = []

    for batch in data_loader:
        # Tokenize the string pairs in the batch
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512, add_special_tokens=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)  # Add attention mask

        # Perform predictions with no gradient calculations
        with torch.no_grad():
            # Pass both input_ids and attention_mask to the model
            outputs = model(input_ids, attention_mask=attention_mask)  
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        
        # Append probabilities for the positive class
        prob_class_positive = probabilities[:, 1].cpu().numpy()
        prob_class_positives.extend(prob_class_positive)

    return np.array(prob_class_positives)

def prepare_and_sort_dataframe(data, probabilities):
    """Prepare the final DataFrame and sort it by probabilities."""
    output = np.column_stack((data, probabilities))
    df = pd.DataFrame(output, columns=['Taxonomy1', 'Taxonomy2', 'Value', 'Probability'])
    df['Probability'] = df['Probability'].astype(float)  # Ensure proper data type for sorting
    df['Value'] = df['Value'].astype(float)
    df_sorted = df.sort_values(by='Value', ascending=True)
    filtered_df = df_sorted.query("Probability > 0.98 & Value < 0.30")
    rest_df = df_sorted.query("not (Probability > 0.98 & Value < 0.30)")
    return filtered_df, rest_df
    

def find_matching_with_LLM(query, target, model, tokenizer, device, batch_size=16):
    """
    Comprehensive function to handle the process from finding nearest neighbors to sorting the data.
    
    :param query: List of query strings.
    :param target: List of target strings.
    :param model: Trained ML model.
    :param tokenizer: Tokenizer corresponding to the ML model.
    :param device: Compute device (CPU or GPU).
    :param batch_size: Batch size for model prediction.
    :return: Tuple of DataFrames (filtered_df, rest_df)
    """
    # Find nearest neighbors
    matches = find_nearest_neighbors(query, target, n_neighbors=3, analyzer_func=ngrams)

    # Prepare data for the model
    data = matches
    tax1 = data[:, 0]
    tax2 = data[:, 1]
    paired_strings = [f"{t1} [SEP] {t2}" for t1, t2 in zip(tax1, tax2)]
    
    # Create DataLoader
    data_loader = create_data_loader(paired_strings, batch_size=batch_size)
    
    # Make predictions
    probabilities = make_predictions(model, data_loader, tokenizer, device)
    
    # Prepare and sort dataframe
    filtered_df, rest_df = prepare_and_sort_dataframe(data, probabilities)

    return filtered_df, rest_df
