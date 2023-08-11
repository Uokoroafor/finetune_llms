# This is a file for the tutorial on how to fine-tune BERT for sentence classification.

# Import the required libraries
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader

# Load a sample dataset
dataset = load_dataset('glue', 'mrpc')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def tokenize_function(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length',
                     max_length=128, return_tensors="pt")


tokenized_datasets = dataset.map(tokenize_function, batched=False)

BATCH_SIZE = 4
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=BATCH_SIZE)

# Load the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Fine-tune the model
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=5e-5)

EPOCHS = 1

for epoch in range(EPOCHS):
    for batch in train_dataloader:
        optimizer.zero_grad()

        # Convert list of tensors to tensors
        input_ids = torch.stack(batch['input_ids']).to(device)
        attention_mask = torch.stack(batch['attention_mask']).to(device)
        labels = torch.stack(batch['label']).to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{EPOCHS} Loss: {loss.item()}")

# Save the model
model.save_pretrained('./my_finetuned_bert')
tokenizer.save_pretrained('./my_finetuned_bert')
