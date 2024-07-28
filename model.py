import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification

class IntentDataset(Dataset):
    def __init__(self, texts, labels, config):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained(config['model_name'])
        self.max_length = config['max_length']

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_model(num_labels):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels).to(device)
    return model