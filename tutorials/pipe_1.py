from transformers import pipeline

# Allocate a pipeline for sentiment-analysis
classifier = pipeline('sentiment-analysis')

# Test some random sentences
print(classifier('We are very happy to show you the huggingface transformers library.'))
print(classifier('We hope you do not hate it.'))
print(classifier('This movie sucks.'))
print(classifier('This movie is great.'))
print(classifier('Your performance was just adequate.'))

# Try text generation
generator = pipeline('text-generation', model='gpt2')
print(generator('Today our aim is to', max_length=30, num_return_sequences=5))

# Try Question Answering
question_answerer = pipeline('question-answering')
