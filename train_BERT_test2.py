import pandas as pd
import torch
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, BertConfig
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup

from utils.train_utils import Trainer

# Preallocate variables defined in set_training_hyperparameters
training_params = dict(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                       epochs=500,
                       batch_size=8,
                       eval_every=1,
                       eval_iters=1,
                       max_seq_len=512,
                       save_every=50, )

learning_params = dict(lr=3e-4, eps=1e-8)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = training_params['max_seq_len']

folder_loc = 'tutorials/'

data = pd.read_csv(folder_loc + 'data/shelfbounce/train_fixed.csv')
val_data = pd.read_csv(folder_loc + 'data/shelfbounce/val_fixed.csv')
test_data = pd.read_csv(folder_loc + 'data/shelfbounce/test_fixed.csv')


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


def decode_data(tokenizer, input_ids):
    return tokenizer.decode(input_ids, skip_special_tokens=True)


# get the questions column and convert to list
texts = data['question'].tolist()
labels = data['answer'].tolist()

val_texts = val_data['question'].tolist()
val_labels = val_data['answer'].tolist()

test_texts = test_data['question'].tolist()
test_labels = test_data['answer'].tolist()

input_ids, attention_masks = encode_data(tokenizer, texts, max_length)
labels = torch.tensor(labels)
dataset = TensorDataset(input_ids, attention_masks, labels)

val_input_ids, val_attention_masks = encode_data(tokenizer, val_texts, max_length)
val_labels = torch.tensor(val_labels)
val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)

test_input_ids, test_attention_masks = encode_data(tokenizer, test_texts, max_length)
test_labels = torch.tensor(test_labels)
test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)

# Create the data loaders


config = BertConfig.from_pretrained("bert-base-uncased")
config.num_labels = 1

model = BertForSequenceClassification(config)

# Freeze the first 8 layers of the model
for param in model.bert.parameters():
    param.requires_grad = False

for param in model.bert.encoder.layer[:8].parameters():
    param.requires_grad = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

batch_size = training_params['batch_size']
dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

optimizer = AdamW(model.parameters(), lr=learning_params['lr'], eps=learning_params['eps'])
epochs = training_params['epochs']
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader) * epochs)
loss_fn = MSELoss()

BertTrainer = Trainer(model=model,
                      optimiser=optimizer,
                      scheduler=scheduler,
                      loss_fn=loss_fn,
                      training_hyperparameters=training_params,
                      tokenizer=tokenizer,
                      )

model, _, _ = BertTrainer.train(
    train_dataloader=dataloader,
    val_dataloader=val_dataloader,
    save_model=True,
    plotting=True,
    verbose=True,
    early_stopping=True,
    early_stopping_patience=10,
)

test_error = BertTrainer.calculate_test_loss(test_data=test_dataloader)
print(f'Test error: {test_error:,.4f}')
