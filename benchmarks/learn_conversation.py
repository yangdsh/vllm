import json
import pandas as pd
import torch
import torch.nn as nn
import time
import sys
sys.setrecursionlimit(15000)  # Increase as needed
import math
from torch.optim import Adam
import numpy as np
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer


def calculate_metrics(y_true, y_pred):
    """Calculate precision, recall, and F1-score."""
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=None)
    return precision, recall, f1

def calculate_label_distribution(predictions):
    unique, counts = np.unique(predictions, return_counts=True)
    total = len(predictions)
    distribution = {label: (count / total) * 100 for label, count in zip(unique, counts)}
    return distribution

class BaselineModel:
    """Baseline model that predicts follow-up probability based only on the overall ratio of follow-ups."""
    
    def __init__(self, df):
        self.df = df
        self.follow_up_ratio = None

    def train(self):
        """Train the baseline by computing the overall probability of a follow-up."""
        self.follow_up_ratio = self.df["follow_up"].mean()
        
        print("\n=== Follow-Up Probability ===")
        print(f"Overall Follow-Up Ratio: {self.follow_up_ratio:.4f}")

    def predict(self, num_samples):
        """Predict follow-up labels by sampling from Bernoulli distribution."""
        return np.random.choice([0, 1], size=num_samples, p=[1 - self.follow_up_ratio, self.follow_up_ratio]).astype(int)


class BaselineModel2:
    """Baseline model that predicts the probability of a follow-up based on historical conversation lengths."""
    
    def __init__(self, df):
        self.df = df
        self.probability_table = None

    def train(self):
        """Train the baseline by computing the probability of continuation and count for different conversation lengths."""
        # Group by 'turns' and calculate the mean and count of 'follow_up'
        group_stats = self.df.groupby("turns")["follow_up"].agg(['mean', 'size']).reset_index()
        group_stats.columns = ['turns', 'follow_up_probability', 'count']

        # Convert the DataFrame to a dictionary for easy access
        self.probability_table = group_stats.set_index('turns')['follow_up_probability'].to_dict()

        # Print the probability distribution with counts
        print("\n=== Follow-Up Probability Distribution ===")
        for _, row in group_stats.iterrows():
            print(f"Rounds: {row['turns']}, Count: {row['count']}, Follow-Up Probability: {row['follow_up_probability']:.4f}")


    def predict(self, turns):
        """Predict whether the conversation will continue based on probability."""
        probability = self.probability_table.get(turns, 0.5)  # Default to 50% if unseen turns
        return 1 if probability >= 0.5 else 0


class MLPClassifier(nn.Module):
    """A configurable Multi-Layer Perceptron classifier for follow-up prediction."""
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=2, num_layers=3, dropout=0.3):
        """
        Args:
            input_dim (int): The input dimension (embedding size).
            hidden_dim (int): The number of units in hidden layers.
            num_layers (int): The number of hidden layers.
            dropout (float): Dropout rate to prevent overfitting.
        """
        super(MLPClassifier, self).__init__()
        layers = []

        # First hidden layer (input to hidden)
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.Dropout(dropout))

        # Additional hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout))

        # Output layer (hidden to 2-class classification)
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class FollowUpPredictor:
    """ML-based model using Sentence Transformers and a multi-layer MLP classifier."""
    
    def __init__(self, dataset_choice, df, bert_model_name, hidden_dim=128, num_layers=3, dropout=0.3):
        self.df = df
        self.bert = SentenceTransformer(bert_model_name)
        self.train_dataloader = None
        self.train_data = None
        self.test_data = None
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.dataset_choice = dataset_choice
        self.task = "classification"
        #if self.dataset_choice == "Tay":
        #    self.task = "regression"

    def prepare_data(self, test_size=0.2):
        """Prepare train-test split and convert data into required format."""
        train_data, test_data = train_test_split(
            self.df, test_size=test_size, random_state=42
        )

        self.train_data = train_data
        self.test_data = test_data

        label_name = "follow_up"
        if self.task == "regression":
            label_name = 'tta'

        train_examples = [
            (row["text"], row["turns"], 
             torch.tensor(row[label_name], dtype=torch.float if self.task == "regression" else torch.long))
            for _, row in train_data.iterrows()
        ]

        self.train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    def train(self, num_epochs=6, lr=2e-5, no_embedding=False):
        """Train using a configurable MLP classifier with CrossEntropyLoss."""
        sentence_embedding_dimension = self.bert.get_sentence_embedding_dimension()

        # Initialize the MLP classifier
        self.classifier = MLPClassifier(
            input_dim=sentence_embedding_dimension + 1,
            hidden_dim=self.hidden_dim,
            output_dim=1 if self.task == "regression" else 2,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.bert.device)
        print(self.bert.device)

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        if self.task == "regression":
            criterion = nn.MSELoss()
        optimizer = Adam(self.classifier.parameters(), lr=lr)

        self.classifier.train()
        best_test_accuracy = 0.0

        for epoch in range(num_epochs):
            total_loss = 0
            start_time = time.time()

            for batch_idx, batch in enumerate(self.train_dataloader):
                input_texts, input_vals, labels = batch  # Unpack tuple
                if self.task == 'regression' and labels.dim() == 1:
                    labels = labels.unsqueeze(1)

                input_texts = [str(text) for text in input_texts]
                labels = labels.clone().detach().to(self.bert.device)

                # Encode sentences into embeddings
                embeddings = self.bert.encode(input_texts, convert_to_tensor=True, batch_size=len(input_texts))
                embeddings *= 0 if no_embedding else 1

                input_vals = input_vals.clone().detach().to(self.bert.device)
                if input_vals.dim() == 1:
                    input_vals = input_vals.unsqueeze(1)  # Expand dims if needed

                # Concatenate embeddings and input_vals along feature dimension (dim=1)
                combined_features = torch.cat((embeddings, input_vals), dim=1)
                
                # Forward pass
                logits = self.classifier(combined_features)

                # Compute loss
                loss = criterion(logits, labels)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Print progress every 100 batches
                if batch_idx % 5000 == 0:
                    elapsed_time = time.time() - start_time
                    estimated_time = (elapsed_time / (batch_idx + 1)) * (len(self.train_dataloader) - batch_idx - 1)
                    print(f"Epoch {epoch+1} [{batch_idx}/{len(self.train_dataloader)}]: "
                          f"Batch Loss: {loss.item():.4f}, Elapsed Time: {elapsed_time:.2f}s, "
                          f"Estimated Time Remaining: {estimated_time:.2f}s")

            print(f"Epoch {epoch+1} completed. Average Loss: {total_loss / len(self.train_dataloader):.4f}")
            self.save_model('', epoch+1)
            y_pred = self.predict(self.test_data)
            y_true = self.test_data["follow_up"].tolist()
            if self.task == "regression":
                y_pred_classifier = np.array(y_pred) > np.mean(y_pred)
            else:
                y_pred_classifier = y_pred
            classifier_precision, classifier_recall, classifier_f1 = calculate_metrics(y_true, y_pred_classifier)
            print(f"Test Precision: {classifier_precision:.4f}, Recall: {classifier_recall:.4f}, F1-score: {classifier_f1}")


    def predict(self, test_data, no_embedding=False):
        """Compute accuracy for the given dataloader."""
        label_name = "follow_up"
        if self.task == "regression":
            label_name = "tta"

        test_examples = [
            (row["text"], row["turns"], torch.tensor(row[label_name], 
            dtype=torch.float if self.task == "regression" else torch.long)) 
            for _, row in test_data.iterrows()
        ]
        test_dataloader = DataLoader(test_examples, shuffle=False, batch_size=16)
        y_true = []
        y_pred = []
        self.classifier.eval()
        total_loss = 0

        for batch in test_dataloader:
            input_texts, input_vals, labels = batch  # Unpack batch
            input_texts = [str(text) for text in input_texts]
            if self.task == 'regression' and labels.dim() == 1:
                labels = labels.unsqueeze(1)

            labels = labels.to(self.bert.device)

            # Encode and predict
            embeddings = self.bert.encode(input_texts, convert_to_tensor=True, batch_size=len(input_texts))
            embeddings *= 0 if no_embedding else 1
            input_vals = input_vals.clone().detach().to(self.bert.device)
            if input_vals.dim() == 1:
                input_vals = input_vals.unsqueeze(1)  # Expand dims if needed

            # Concatenate embeddings and input_vals along feature dimension (dim=1)
            combined_features = torch.cat((embeddings, input_vals), dim=1)
            logits = self.classifier(combined_features)
            criterion = nn.CrossEntropyLoss()
            if self.task == "regression":
                criterion = nn.MSELoss()
            loss = criterion(logits, labels)
            total_loss += loss.item()
            if self.task == "regression":
                predictions = logits.detach().cpu().numpy()
            else:
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1).cpu().numpy()

            y_true.extend(labels)
            y_pred.extend(predictions)
        #for i in range(len(input_texts)):
        #    print(probabilities[i], predictions[i], labels[i])
        print(f'Test Loss: {total_loss / len(test_dataloader):.4f}')
        return y_pred

    def save_model(self, save_path, epoch):
        """Save the trained model and classifier."""
        # self.classifier.save(save_path)
        classifier_path = f"{save_path}{self.dataset_choice}{epoch}.pt"
        torch.save(self.classifier.state_dict(), classifier_path)
        print(f"Classifier saved to {classifier_path}!")

def combine_user_requests(messages):
    # Gather up to 5 most recent user messages (including the current one)
    user_msgs = []
    j = len(messages) - 1
    value_tag = 'content'
    if value_tag not in messages[0]:
        value_tag = 'value'
    while j >= 0 and len(user_msgs) < 5:
        user_msgs.append(messages[j][value_tag])
        j -= 2
    user_msgs.reverse()  # Order messages chronologically

    # Determine allowed words per message (total must not exceed 512 words)
    n = len(user_msgs)
    allowed_words_per_message = 512 // n if n > 0 else 0

    processed_msgs = []
    for msg in user_msgs:
        words = msg.split()
        if len(words) > allowed_words_per_message:
            # Determine the split point: first half and last half of allowed words
            first_half_count = allowed_words_per_message // 2
            second_half_count = allowed_words_per_message - first_half_count
            first_part = words[:first_half_count]
            second_part = words[-second_half_count:]
            processed_msg = " ".join(first_part + second_part)
        else:
            processed_msg = msg
        processed_msgs.append(processed_msg)

    # Concatenate the processed user messages with the phrase in between
    combined_text = " [new message]: ".join(processed_msgs)

    return combined_text

def format_text(combined_text, uuid):
    combined_text = " [user id]: " + uuid + ". " + combined_text
    combined_text = "query: " + combined_text
    return combined_text

def load_conv_data(data_source, N, format):
    """Load and preprocess conversation data from either a JSON file or Hugging Face dataset."""
    
    conversation_features = []
    first_messages = []
    first_messages_raw = []
    first_responses = []
    tokenizer = get_tokenizer("Qwen/Qwen2.5-0.5B",
                              tokenizer_mode="auto",
                              trust_remote_code=True)

    if isinstance(data_source, str):  # Loading from a local JSON file
        print(f"Loading dataset from JSON file: {data_source}")
        with open(data_source, 'r', encoding='utf-8') as f:
            data = json.load(f)

    elif isinstance(data_source, dict):  # Hugging Face dataset case
        print("Loading dataset from Hugging Face Dataset...")
        data = data_source["train"].to_pandas().to_dict(orient="records")  # Convert dataset to list of dicts

    else:
        raise ValueError("Invalid data source! Must be a JSON file path or Hugging Face dataset.")

    uuid_mapping = {}
    
    for convo in data[:N]:
        messages = convo.get("conversation", []) #lmsys
        role = "role"
        user = "user"
        value_tag = 'content' 
        if len(messages) == 0:
            messages = convo.get("conversation_a", [])
        if len(messages) == 0: #sharegpt
            role = "from"
            user = "human"
            messages = convo.get("conversations", [])
            value_tag = 'value'
        turns = 0  # Initialize turns at the start of each conversation

        if len(messages) > 1:
            prompt_tokens = tokenizer(messages[0][value_tag]).input_ids
            response_tokens = tokenizer(messages[1][value_tag]).input_ids
            if len(prompt_tokens) < 16384:
                first_messages.append(prompt_tokens)
                first_messages_raw.append(messages[0][value_tag])
                first_responses.append(response_tokens)

        for i in range(len(messages) - 1):
            if messages[i][role] == user and messages[i + 1][role] in ("assistant", "gpt"):


                # (Optionally, you can append the assistant_response here if needed)
                # For example: combined_text += f" [assistant]: {assistant_response}"

                follow_up = 1 if (i + 2 < len(messages) and messages[i + 2][role] == user) else 0

                combined_text = combine_user_requests(messages[:i+1])
                conversation_features.append({
                    "follow_up": follow_up,
                    "turns": turns  # Ensure turns is correctly stored
                })
                if 'timestamp' in messages[i]:
                    tta = 24*3600
                    if follow_up:
                        tta = messages[i+2]['timestamp'] - messages[i]['timestamp']
                    conversation_features[-1]['tta'] = tta
                    conversation_features[-1]['tta'] = math.log(tta + 1)
                if 'uuid' in messages[i]:
                    uuid = messages[i]['uuid']
                    if uuid not in uuid_mapping:
                        uuid_mapping[uuid] = len(uuid_mapping)
                    conversation_features[-1]['uuid'] = uuid_mapping[uuid]
                    uuid_mapped = str(conversation_features[-1]['uuid'])
                else:
                    uuid_mapped = ''
                conversation_features[-1]['text'] = format_text(combined_text, uuid_mapped)

                turns += 1  # Increment for the next pair

    df = pd.DataFrame(conversation_features)

    def build_trie_with_freq(messages):
        """
        Build a trie from the tokenized messages.
        Each node in the trie is represented as a dictionary with two keys:
        - 'freq': an integer counting how many times the token appears
        - 'children': a dictionary for the next tokens
        """
        trie = {}
        cnt = 0
        for message in messages:
            tokens = message  # Basic tokenization; adjust as needed.
            node = trie
            for token in tokens:
                if token not in node:
                    node[token] = {'freq': 0, 'children': {}}
                node[token]['freq'] += 1
                node = node[token]['children']
                cnt += 1
        return trie, cnt

    def print_trie_levels(trie, level=1, max_level=5, indent=""):
        """
        Recursively print the trie up to max_level.
        
        Args:
            trie (dict): The trie (or subtree) to print.
            level (int): Current level (starting at 1).
            max_level (int): Maximum level to print.
            indent (str): Used for pretty-printing with indentation.
        """
        if level > max_level:
            return
        for token, node in trie.items():
            if node['freq'] > 20:
                print(f"{indent}{token} (freq: {node['freq']})")
                print_trie_levels(node['children'], level+1, max_level, indent + "  ")

    def count_nodes(trie):
        """
        Recursively count the number of nodes (tokens) in the trie.
        
        Args:
            trie (dict): A trie represented as a nested dictionary.
            
        Returns:
            int: Total number of nodes in the trie.
        """
        count = 0
        freq = 0
        for token, node in trie.items():
            freq += node['freq']
            subtree = node['children']
            count += 1  # count this token
            res = count_nodes(subtree)
            count += res[0]
            freq += res[1]
        return count, freq


    trie, cnt = build_trie_with_freq(first_messages)
    print(cnt)
    # print_trie_levels(trie)
    unique_prefix_tokens_count, all_tokens = count_nodes(trie)
    print("Prefix tokens count (shared/all):", all_tokens - unique_prefix_tokens_count, '/', all_tokens)
    print((all_tokens - unique_prefix_tokens_count) / all_tokens)
    print("Response tokens count (all):", sum([len(message) for message in first_responses]))
    print(f"Number of all requests: {len(first_messages)}")
    unique_messages = set(first_messages_raw)
    print(f"Number of duplicated requests: {len(first_messages) - len(unique_messages)}")
    total_length_duplicates = 0
    for i, msg in enumerate(first_messages_raw):
        if msg not in unique_messages or unique_messages.remove(msg):
            total_length_duplicates += len(first_messages[i])
    print("dup tokens: ", total_length_duplicates)

    print("\n=== DataFrame Columns ===")
    print(df.columns)

    return df


bert_model_name="sentence-transformers/all-MiniLM-L6-v2"
bert_model_name="intfloat/multilingual-e5-small"
hidden_dim=256
num_layers=3

# ==== RUN MODEL ====
if __name__ == "__main__":
    dataset_choice = "sharegpt" #"sharegpt" # "lmsys-chat-1m"

    if dataset_choice == "lmsys-chat-1m":
        print("Loading dataset: LMSys-chat-1M from Hugging Face...")
        ds = load_dataset("lmsys/lmsys-chat-1m")  # Use authentication if required
        df = load_conv_data(ds, 100000, 'lmsys')
        print("\n=== Sample Data from Processed Dataset ===")
        print(df.head())
    elif dataset_choice == "sharegpt":
        print(f"Loading dataset: ShareGPT")
        df = load_conv_data("../../ShareGPT_V3_unfiltered_cleaned_split.json", 100000, 'sharegpt')
    elif dataset_choice == "Tay":
        print(f"Loading dataset: Tay from")
        df = load_conv_data("../../tay.json", 100000, 'sharegpt')
    elif dataset_choice == "chatbot_arena":
        print("Loading dataset: Chatbot Arena Conversations from Hugging Face...")
        ds = load_dataset("lmsys/chatbot_arena_conversations")
        df = load_conv_data(ds, 100000, 'chatbot_arena')
    else:
        raise ValueError("Invalid dataset choice! Choose between 'lmsys-chat-1m' and 'sharegpt'.")

    # Run Baseline Model
    baseline = BaselineModel(df)
    baseline.train()

    baseline2 = BaselineModel2(df)
    baseline2.train()

    # Initialize and train model
    predictor = FollowUpPredictor(dataset_choice, df, bert_model_name, hidden_dim=hidden_dim, num_layers=num_layers, dropout=0.)
    predictor.prepare_data()
    # predictor.train(num_epochs=6, lr=5e-5)
    predictor.train(num_epochs=6, lr=5e-4)

    test_df = predictor.test_data
    y_true = test_df["follow_up"].tolist()
    # Get predictions from the baseline model
    y_pred_baseline = baseline.predict(len(y_true))
    y_pred_baseline2 = [baseline2.predict(turns) for turns in test_df["turns"]]
    y_pred_classifier = predictor.predict(test_df)

    # Compute metrics for Baseline Model
    baseline_precision, baseline_recall, baseline_f1 = calculate_metrics(y_true, y_pred_baseline)
    print(f"\nBaseline Model Metrics:")
    print(f"Precision: {baseline_precision:.4f}, Recall: {baseline_recall:.4f}, F1-score: {baseline_f1}")
    baseline_precision, baseline_recall, baseline_f1 = calculate_metrics(y_true, y_pred_baseline2)
    print(f"\nBaseline2 Model Metrics:")
    print(f"Precision: {baseline_precision:.4f}, Recall: {baseline_recall:.4f}, F1-score: {baseline_f1}")

    # Calculate and print label distribution for the classifier
    # classifier_distribution = calculate_label_distribution(y_pred_baseline)
    # print("baseline Prediction Distribution:", classifier_distribution)
    
    # Calculate and print label distribution for the classifier
    # classifier_distribution = calculate_label_distribution(y_pred_classifier)
    # print("Classifier Prediction Distribution:", classifier_distribution)
else:
    class Predictor:
        def __init__(self, model_path, bert_model):
            self.bert = bert_model
            self.device = self.bert.device
            self.classifier = MLPClassifier(
                input_dim=self.bert.get_sentence_embedding_dimension() + 1,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=0.
            ).to(self.device)
            self.classifier.load_state_dict(torch.load(model_path, map_location=self.device))
            self.classifier.eval()

        def predict_prob(self, conv, turns, no_embedding=False):
            text = format_text(combine_user_requests(conv), '')
            embeddings = self.bert.encode([text], convert_to_tensor=True, batch_size=1)
            embeddings *= 0 if no_embedding else 1

            input_vals = torch.tensor([[turns]], dtype=torch.float).to(self.device)
            combined_features = torch.cat((embeddings, input_vals), dim=1)

            with torch.no_grad():
                logits = self.classifier(combined_features)
                probabilities = torch.softmax(logits, dim=1)

            prob_has_next = probabilities[0][1].item()  # probability for "having next turn"

            # print(text, turns)

            return prob_has_next

    def make_predictor(checkpoint):
        predictor_instance = Predictor(checkpoint, SentenceTransformer(bert_model_name))
        return predictor_instance