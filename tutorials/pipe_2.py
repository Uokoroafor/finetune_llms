from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Now let's test it while specifying the model and tokenizer
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


# Allocate a pipeline for sentiment-analysis
classifier = pipeline('sentiment-analysis')
res = classifier('We are very happy to show you the huggingface transformers library.')
print(res)


classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
res = classifier('We are very happy to show you the huggingface transformers library.')
print(res)

# Answer should be the same

# Test tokeniser
statement = 'We are very happy to show you the huggingface transformers library.'
tokens = tokenizer(statement)
print(tokens)
ids = tokens['input_ids'] # tokenizer.convert_tokens_to_ids(tokens)
print(ids)

ids2 = tokenizer.convert_ids_to_tokens(ids)
print(ids2)

decoded = tokenizer.decode(ids)
print(decoded)
