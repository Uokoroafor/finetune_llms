from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import pandas as pd

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 512  # Change according to your needs

data = pd.read_csv('data/shelfbounce/train_fixed.csv')
val_data = pd.read_csv('data/shelfbounce/val_fixed.csv')
test_data = pd.read_csv('data/shelfbounce/test_fixed.csv')


def encode_data(tokenizer, texts, max_length):
    input_ids = []
    attention_masks = []

    for text in texts:
        encode = tokenizer.encode_plus(text, add_special_tokens=True,
                                       max_length=max_length, truncation=True, padding='max_length',
                                       return_tensors='pt')
        input_ids.append(encode['input_ids'][0])
        attention_masks.append(encode['attention_mask'][0])

    return torch.stack(input_ids), torch.stack(attention_masks)


# get the questions column and convert to list
texts = data['question'].tolist()
labels = data['answer'].tolist()

input_ids, attention_masks = encode_data(tokenizer, texts, max_length)
labels = torch.tensor(labels)
dataset = TensorDataset(input_ids, attention_masks, labels)

print(dataset[0])

# Create the data loaders

from transformers import BertForSequenceClassification, BertConfig

config = BertConfig.from_pretrained("bert-base-uncased")
config.num_labels = 1

model = BertForSequenceClassification(config)

from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn import MSELoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the classifier layer
for param in model.classifier.parameters():
    param.requires_grad = True


batch_size = 8
dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=batch_size)
optimizer = AdamW(model.parameters(), lr=3e-5, eps=1e-8)
epochs = 3
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader) * epochs)

loss_fn = MSELoss()

model.train()
for epoch in range(epochs):
    for batch in dataloader:
        input_ids = batch[0].to(device)
        attention_masks = batch[1].to(device)
        labels = batch[2].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_masks)[0].squeeze()
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

# Save the model
model.save_pretrained("./my_bert_regression_model/")
tokenizer.save_pretrained("./my_bert_regression_model/")

