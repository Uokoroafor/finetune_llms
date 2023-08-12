from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import pandas as pd
import time

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 512  # Change according to your needs

folder_loc = 'tutorials/d'

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

# print(dataset[0])

# Create the data loaders

from transformers import BertForSequenceClassification, BertConfig

config = BertConfig.from_pretrained("bert-base-uncased")
config.num_labels = 1

model = BertForSequenceClassification(config)

from transformers import get_linear_schedule_with_warmup
from torch.nn import MSELoss
from torch.optim import AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
# # Freeze all parameters
# for param in model.parameters():
#     param.requires_grad = False
#
# # Unfreeze the classifier layer
# for param in model.classifier.parameters():
#     param.requires_grad = True

batch_size = 16
dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

optimizer = AdamW(model.parameters(), lr=3e-5, eps=1e-8)
epochs = 20
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader) * epochs)

loss_fn = MSELoss()

model.train()
start_time = time.time()
for epoch in range(epochs):
    epoch_start = time.time()
    epoch_loss = 0
    for idx, batch in enumerate(dataloader):
        input_ids = batch[0].to(device)
        attention_masks = batch[1].to(device)
        labels = batch[2].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_masks)[0].squeeze()
        loss = loss_fn(outputs, labels)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        # print every 100 steps
        # if (idx + 1) % 100 == 0:
        #     print(f'Batch_ID: {idx + 1}, Epoch: {epoch + 1}, Loss:  {loss.item():,.4f}')

    epoch_end = time.time()

    hours = (epoch_end - epoch_start) // 3600
    minutes = ((epoch_end - epoch_start) % 3600) // 60
    seconds = (epoch_end - epoch_start) % 60
    print(f'Epoch{epoch + 1} complete in {hours} hour(s), {minutes} minute(s) and {seconds: .2f} seconds. Loss: {epoch_loss / (idx + 1):,.4f}')
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f'./my_bert_regression_model/epoch{epoch + 1}.pth')
        # Validation
        model.eval()
        val_loss = 0
        for idx_, batch in enumerate(val_dataloader):
            input_ids = batch[0].to(device)
            attention_masks = batch[1].to(device)
            labels = batch[2].to(device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_masks)[0].squeeze()
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

        print(f'At Epoch {epoch}: Validation Loss: {val_loss / (idx_ + 1):,.4f} - Training Loss: {epoch_loss / (idx + 1):,.4f}')


# Total training time
end_time = time.time()
hours = (end_time - start_time) // 3600
minutes = ((end_time - start_time) % 3600) // 60
seconds = (end_time - start_time) % 60
print(f'Training complete in {hours} hour(s), {minutes} minute(s) and {seconds: .2f} seconds.')


# Save the model
model.save_pretrained("./my_bert_regression_model/")
tokenizer.save_pretrained("./my_bert_regression_model/")
print("Model saved!")
