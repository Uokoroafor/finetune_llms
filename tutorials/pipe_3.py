from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import os

# Now let's test it while specifying the model and tokenizer
# If model exists, load it, otherwise download it
if os.path.exists('models/sentiment/'):
    print('Loading model from models/sentiment')
    model = AutoModelForSequenceClassification.from_pretrained('models/sentiment')
    tokenizer = AutoTokenizer.from_pretrained('models/sentiment')

else:
    print('Downloading model')
    model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

X_train = ["Verily I say unto thee, that this is a very good movie.", "The wheels are in motion", "The die is cast",
           "Things fall apart", "The center cannot hold", "The best lack all conviction",
           "The worst are full of passionate intensity", "Surely some revelation is at hand"]

res = classifier(X_train)

print(res)

batch = tokenizer(X_train, padding=True, truncation=True, return_tensors="pt", max_length=512)
print(batch)

with torch.no_grad():
    logits = model(**batch)[0]
    probs = F.softmax(logits, dim=-1)
    print(probs)
    labels = torch.argmax(probs, dim=-1)
    print(labels)


# Save the model and tokenizer
model.save_pretrained('models/sentiment')
tokenizer.save_pretrained('models/sentiment')