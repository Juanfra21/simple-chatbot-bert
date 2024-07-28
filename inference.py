import torch
import ast
import random
from transformers import BertTokenizer

def get_response(text, model, intent_to_index, df_response):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=50,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask)

    _, predicted = torch.max(output.logits, 1)
    intent_index = predicted.item()
    
    index_to_intent = {v: k for k, v in intent_to_index.items()}
    predicted_intent = index_to_intent[intent_index]

    response_str = df_response[df_response['intent'] == predicted_intent]['response'].iloc[0]
    response_list = ast.literal_eval(response_str)
    response = random.choice(response_list)

    return response, predicted_intent