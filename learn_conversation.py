import json
import pandas as pd
import torch
import torch.nn as nn
import time
from torch.optim import Adam
import numpy as np
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

def simulate_label_removal(df, remove_percentage):
    """
    Simulate removing x% of samples where label = 1 and return the modified dataset.
    """
    df_1 = df[df["follow_up"] == 1]  # Samples with label 1
    df_0 = df[df["follow_up"] == 0]  # Samples with label 0

    num_to_remove = int(len(df_1) * (remove_percentage / 100.0))
    df_1_remaining = df_1.sample(n=len(df_1) - num_to_remove, random_state=42)

    df_modified = pd.concat([df_0, df_1_remaining]).sample(frac=1, random_state=42).reset_index(drop=True)
    return df_modified

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
        # Group by 'num_pairs' and calculate the mean and count of 'follow_up'
        group_stats = self.df.groupby("num_pairs")["follow_up"].agg(['mean', 'size']).reset_index()
        group_stats.columns = ['num_pairs', 'follow_up_probability', 'count']

        # Convert the DataFrame to a dictionary for easy access
        self.probability_table = group_stats.set_index('num_pairs')['follow_up_probability'].to_dict()

        # Print the probability distribution with counts
        print("\n=== Follow-Up Probability Distribution ===")
        for _, row in group_stats.iterrows():
            print(f"Rounds: {row['num_pairs']}, Count: {row['count']}, Follow-Up Probability: {row['follow_up_probability']:.4f}")


    def predict(self, num_pairs):
        """Predict whether the conversation will continue based on probability."""
        probability = self.probability_table.get(num_pairs, 0.5)  # Default to 50% if unseen num_pairs
        return 1 if probability >= 0.5 else 0


class MLPClassifier(nn.Module):
    """A configurable Multi-Layer Perceptron classifier for follow-up prediction."""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.3):
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
        layers.append(nn.Linear(hidden_dim, 2))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class FollowUpPredictor:
    """ML-based model using Sentence Transformers and a multi-layer MLP classifier."""
    
    def __init__(self, df, model_name="sentence-transformers/all-MiniLM-L6-v2", hidden_dim=128, num_layers=3, dropout=0.3):
        self.df = df
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.train_dataloader = None
        self.test_dataloader = None
        self.train_data = None
        self.test_data = None
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

    def prepare_data(self, test_size=0.2):
        """Prepare train-test split and convert data into required format."""
        train_data, test_data = train_test_split(
            self.df, test_size=test_size, random_state=42, stratify=self.df["follow_up"]
        )

        self.train_data = train_data
        self.test_data = test_data

        train_examples = [
            (row["text"], row["num_pairs"], torch.tensor(row["follow_up"], dtype=torch.long)) for _, row in train_data.iterrows()
        ]

        self.train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    def train(self, num_epochs=6, lr=2e-5):
        """Train using a configurable MLP classifier with CrossEntropyLoss."""
        sentence_embedding_dimension = self.model.get_sentence_embedding_dimension()

        # Initialize the MLP classifier
        self.classifier = MLPClassifier(
            input_dim=sentence_embedding_dimension + 1,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.model.device)

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.classifier.parameters(), lr=lr)

        self.model.train()
        best_test_accuracy = 0.0

        for epoch in range(num_epochs):
            total_loss = 0
            start_time = time.time()

            for batch_idx, batch in enumerate(self.train_dataloader):
                input_texts, input_vals, labels = batch  # Unpack tuple

                input_texts = [str(text) for text in input_texts]
                labels = torch.tensor(labels, dtype=torch.long).to(self.model.device)

                # Encode sentences into embeddings
                embeddings = self.model.encode(input_texts, convert_to_tensor=True, batch_size=len(input_texts))
                embeddings *= 0

                input_vals = torch.tensor(input_vals, dtype=torch.float).to(self.model.device)
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
                if batch_idx % 100 == 0:
                    elapsed_time = time.time() - start_time
                    estimated_time = (elapsed_time / (batch_idx + 1)) * (len(self.train_dataloader) - batch_idx - 1)
                    print(f"Epoch {epoch+1} [{batch_idx}/{len(self.train_dataloader)}]: "
                          f"Batch Loss: {loss.item():.4f}, Elapsed Time: {elapsed_time:.2f}s, "
                          f"Estimated Time Remaining: {estimated_time:.2f}s")

            print(f"Epoch {epoch+1} completed. Average Loss: {total_loss / len(self.train_dataloader):.4f}")
            y_pred_classifier = predictor.predict(self.test_data)
            y_true = self.test_data["follow_up"].tolist()
            classifier_precision, classifier_recall, classifier_f1 = calculate_metrics(y_true, y_pred_classifier)
            print(f"\nClassifier Model Metrics:")
            print(f"Precision: {classifier_precision:.4f}, Recall: {classifier_recall:.4f}, F1-score: {classifier_f1}")


    def predict(self, test_data):
        """Compute accuracy for the given dataloader."""
        test_examples = [
            (row["text"], row["num_pairs"], torch.tensor(row["follow_up"], dtype=torch.long)) for _, row in test_data.iterrows()
        ]
        test_dataloader = DataLoader(test_examples, shuffle=False, batch_size=512)
        y_true = []
        y_pred = []
        self.classifier.eval()

        for batch in test_dataloader:
            input_texts, input_vals, labels = batch  # Unpack batch
            input_texts = [str(text) for text in input_texts]

            labels = labels.clone().detach().cpu().numpy()

            # Encode and predict
            embeddings = self.model.encode(input_texts, convert_to_tensor=True, batch_size=len(input_texts))
            embeddings *= 0
            input_vals = input_vals.clone().detach().to(self.model.device)
            if input_vals.dim() == 1:
                input_vals = input_vals.unsqueeze(1)  # Expand dims if needed

            # Concatenate embeddings and input_vals along feature dimension (dim=1)
            combined_features = torch.cat((embeddings, input_vals), dim=1)
            logits = self.classifier(combined_features)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()

            y_true.extend(labels)
            y_pred.extend(predictions)
        return y_pred

    def save_model(self, save_path):
        """Save the trained model and classifier."""
        self.model.save(save_path)
        classifier_path = f"{save_path}/classifier.pt"
        torch.save(self.classifier.state_dict(), classifier_path)
        print(f"Classifier saved to {classifier_path}!")


def load_conv_data(data_source):
    """Load and preprocess conversation data from either a JSON file or Hugging Face dataset."""
    
    conversation_features = []

    if isinstance(data_source, str):  # Loading from a local JSON file
        print(f"Loading dataset from JSON file: {data_source}")
        with open(data_source, 'r', encoding='utf-8') as f:
            data = json.load(f)

    elif isinstance(data_source, dict):  # Hugging Face dataset case
        print("Loading dataset from Hugging Face Dataset...")
        data = data_source["train"].to_pandas().to_dict(orient="records")  # Convert dataset to list of dicts

    else:
        raise ValueError("Invalid data source! Must be a JSON file path or Hugging Face dataset.")

    for convo in data[:100000]:
        messages = convo.get("conversation", [])
        role = "role"
        user = "user"
        content = "content"
        if len(messages) == 0:
            messages = convo.get("conversation_a", [])
        if len(messages) == 0:
            role = "from"
            user = "human"
            content = "value"
            messages = convo.get("conversations", [])
        num_pairs = 0  # Initialize num_pairs at the start of each conversation

        for i in range(len(messages) - 1):
            if messages[i][role] == user and messages[i + 1][role] in ("assistant", "gpt"):
                user_message = messages[i][content]
                assistant_response = messages[i + 1][content]
                combined_text = f"User: {user_message} Assistant: {assistant_response}"
                follow_up = 1 if (i + 2 < len(messages) and messages[i + 2][role] == user) else 0

                conversation_features.append({
                    "text": combined_text,
                    "follow_up": follow_up,
                    "num_pairs": num_pairs  # Ensure num_pairs is correctly stored
                })

                num_pairs += 1  # Increment for the next pair

    df = pd.DataFrame(conversation_features)

    print("\n=== DataFrame Columns ===")
    print(df.columns)

    return df



# ==== RUN MODEL ====
if __name__ == "__main__":
    dataset_choice = "lmsys-chat-1m"
    json_file = "../ShareGPT_V3_unfiltered_cleaned_split.json" if dataset_choice == "sharegpt" else None

    if dataset_choice == "lmsys-chat-1m":
        print("Loading dataset: LMSys-chat-1M from Hugging Face...")
        ds = load_dataset("lmsys/lmsys-chat-1m")  # Use authentication if required
        df = load_conv_data(ds)
        print("\n=== Sample Data from Processed Dataset ===")
        print(df.head())
    elif dataset_choice == "sharegpt":
        print(f"Loading dataset: ShareGPT from {json_file}...")
        df = load_conv_data(json_file)
    elif dataset_choice == "chatbot_arena":
        print("Loading dataset: Chatbot Arena Conversations from Hugging Face...")
        ds = load_dataset("lmsys/chatbot_arena_conversations")
        df = load_conv_data(ds)
    else:
        raise ValueError("Invalid dataset choice! Choose between 'lmsys-chat-1m' and 'sharegpt'.")

    # Run Baseline Model
    baseline = BaselineModel(df)
    baseline.train()

    baseline2 = BaselineModel2(df)
    baseline2.train()

    # Initialize and train model
    predictor = FollowUpPredictor(df, hidden_dim=256, num_layers=6, dropout=0.25)
    predictor.prepare_data()
    predictor.train(num_epochs=3, lr=5e-5)
    predictor.save_model(dataset_choice)

    test_df = predictor.test_data
    y_true = test_df["follow_up"].tolist()
    # Get predictions from the baseline model
    y_pred_baseline = baseline.predict(len(y_true))
    y_pred_baseline2 = [baseline2.predict(num_pairs) for num_pairs in test_df["num_pairs"]]
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