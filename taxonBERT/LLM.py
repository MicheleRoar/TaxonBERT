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
    

def find_matching_with_LLM(query_dataset, target_dataset, model, tokenizer, device, batch_size=16):
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

    query_list = list(query_dataset.gbif_taxonomy)
    target_list = list(target_dataset.ncbi_target_string)

    matches = find_nearest_neighbors(query_list, target_list, n_neighbors=3, analyzer_func=ngrams)

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


    df2 = target_dataset.merge(filtered_df, left_on='ncbi_target_string', right_on='Taxonomy2', how='inner')
    df3 = query_dataset.merge(df2, left_on='gbif_taxonomy', right_on='Taxonomy1', how='inner')[['taxonID', 'parentNameUsageID', 'canonicalName', 'ncbi_id', 'ncbi_canonicalName', 'Probability', 'Value', 'taxonomicStatus', 'gbif_taxonomy', 'ncbi_target_string', 'ncbi_lineage_names', 'ncbi_lineage_ids']]

    initial = set(query_list)
    matched = set(df3.gbif_taxonomy)
    discarded = list(initial.difference(matched))

    df_matched = df3.copy()
    ncbi_matching = list(set(df_matched.ncbi_id))
    ncbi_missing = target_dataset[~target_dataset.ncbi_id.isin(ncbi_matching)]
    ncbi_missing_2 = ncbi_missing[['ncbi_id', 'ncbi_canonicalName', 'ncbi_target_string', 'ncbi_lineage_names', 'ncbi_lineage_ids']]
    ncbi_missing_3 = target_dataset[target_dataset['ncbi_canonicalName'].str.contains(r'\d')]
    new_df_matched = pd.concat([df_matched, ncbi_missing_2, ncbi_missing_3], ignore_index=True)
    new_df_matched = new_df_matched.fillna(-1)
    df_unmatched = query_dataset[query_dataset["gbif_taxonomy"].isin(discarded)]

    return (new_df_matched, df_unmatched)


def match_dataset_with_LLM(query_dataset, target_dataset, model, tokenizer, device, tree_generation = False):
    """
    Filters the matched dataset to identify and separate synonyms.

    Args:
    [Your existing parameters]

    Returns:
    tuple: DataFrames of filtered synonyms and unmatched entries.
    """

    df_matched, df_unmatched = find_matching_with_LLM(query_dataset, target_dataset, model, tokenizer, device, batch_size=16)

    # Filter rows where canonicalName is identical to ncbi_canonicalName
    identical = df_matched.query("canonicalName == ncbi_canonicalName")

    # Filter rows where canonicalName is not identical to ncbi_canonicalName
    not_identical = df_matched.query("(canonicalName != ncbi_canonicalName) and taxonID != -1")

    # Filter rows where canonicalName is not identical to ncbi_canonicalName
    only_ncbi = df_matched.query("(taxonID == -1) and ncbi_lineage_ranks != -1")

    matching_synonims = []
    excluded_data = []

    # Pre-elaborazione: converti i nomi in minuscolo una sola volta
    not_identical_ = not_identical.copy()
    not_identical_[['canonicalName', 'ncbi_canonicalName']] = not_identical_[['canonicalName', 'ncbi_canonicalName']].apply(lambda x: x.str.lower())

    """

    # Define columns of interest
    columns_of_interest = ['taxonID', 'parentNameUsageID', 'acceptedNameUsageID', 'canonicalName', 'taxonRank', 'taxonomicStatus', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus']
    gbif_dataset = pd.read_csv("./GBIF_output/Taxon.tsv", sep="\t", usecols=columns_of_interest, on_bad_lines='skip', low_memory=False)

    # Load GBIF dictionary
    gbif_synonyms_names, gbif_synonyms_ids, gbif_synonyms_ids_to_ids = get_gbif_synonyms(gbif_dataset)

    # Load NCBI dictionary
    ncbi_synonyms_names, ncbi_synonyms_ids = get_ncbi_synonyms("./NCBI_output/names.dmp")

    gbif_synonyms_lower = {k.lower(): {v.lower() for v in vs} for k, vs in gbif_synonyms_names.items()}
    ncbi_synonyms_lower = {k.lower(): {v.lower() for v in vs} for k, vs in ncbi_synonyms_names.items()}

    for index, row in not_identical_.iterrows():
        gbif_canonicalName = row['canonicalName']
        ncbi_canonicalName = row['ncbi_canonicalName']

        # Utilizza insiemi per il confronto dei sinonimi
        gbif_synonyms_set = gbif_synonyms_lower.get(gbif_canonicalName, set())
        ncbi_synonyms_set = ncbi_synonyms_lower.get(ncbi_canonicalName, set())

        if gbif_canonicalName in ncbi_synonyms_set or ncbi_canonicalName in gbif_synonyms_set or gbif_synonyms_set & ncbi_synonyms_set:
            matching_synonims.append(row)
        else:
            excluded_data.append(row)
    """

    excluded_data = [] #to remove

    # Converti le liste in DataFrame solo dopo il ciclo
    excluded_data_df = pd.DataFrame(excluded_data)
    doubtful = excluded_data_df.copy()

    if not doubtful.empty:
        # Calculate Levenshtein distance for non-identical pairs
        lev_dist = doubtful.apply(lambda row: Levenshtein.distance(row['canonicalName'], row['ncbi_canonicalName']), axis=1)

        # Create a copy of the filtered DataFrame for non-identical pairs
        similar_pairs = doubtful.copy()

        # Add the Levenshtein distance as a new column
        similar_pairs["levenshtein_distance"] = lev_dist

        possible_typos_df = pd.DataFrame(similar_pairs).query("levenshtein_distance <= 3").sort_values('score')

        gbif_excluded = query_dataset[query_dataset.taxonID.isin(excluded_data_df.taxonID)]
        ncbi_excluded = target_dataset[target_dataset.ncbi_id.isin(excluded_data_df.ncbi_id)]
    else:
        possible_typos_df = "No possible typos detected"


    # Create separate DataFrame for included and excluded data
    df_matching_synonims = pd.DataFrame(matching_synonims).drop_duplicates()
    df_matching_synonims.loc[:, 'ncbi_id'] = df_matching_synonims['ncbi_id'].astype(int)



    # Assuming you have your sets defined
    iden = set(identical.ncbi_id)

    # Filter out the excluded IDs from other DataFrames
    ncbi_excluded_filtered = ncbi_excluded[~ncbi_excluded.ncbi_id.isin(iden)]


    if tree_generation and not doubtful.empty:
        # Concatenate similar pairs with identical samples
        matched_df = pd.concat([identical , df_matching_synonims, only_ncbi, ncbi_excluded_filtered])
    else:
        matched_df = pd.concat([identical , df_matching_synonims])

    matched_df = matched_df.infer_objects(copy=False).fillna(-1)
    matched_df['taxonID'] = matched_df['taxonID'].astype(int)

    if not doubtful.empty:
        # Extract the "gbif_taxonomy" strings from non-similar pairs
        unmatched_df = pd.concat([df_unmatched, gbif_excluded])
    else:
        unmatched_df = df_unmatched

    unmatched_df = unmatched_df.infer_objects(copy=False).fillna(-1)

    matched_df = matched_df.replace([-1, '-1'], None)
    unmatched_df = unmatched_df.replace([-1, '-1'], None)
    return matched_df, unmatched_df, possible_typos_df
