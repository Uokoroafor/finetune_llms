from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering, BertForQuestionAnswering, BertTokenizer
import torch
import torch.nn.functional as F
import os
# import ssl
import pandas as pd

# ssl._create_default_https_context = ssl._create_unverified_context

# Steps for fine-tuning a model
# 1. Load the model and tokenizer
# 2. Load the dataset
# 3. Preprocess the dataset
# 4. Train the model
# 5. Evaluate the model
# 6. Save the model and tokenizer

# Want to finetune a pre-trained model for question answering
# Load models from torch instead of transformers

# model_name = 'deepset/roberta-base-squad2'
model_name = 'bert-base-uncased'
# model = torch.hub.load('huggingface/pytorch-transformers', 'modelForQuestionAnswering', model_name)
# tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# model = AutoModelForQuestionAnswering.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer_params = {'return_tensors': 'pt', 'max_length': 128, 'truncation': True, 'padding': 'max_length'}
# assert model.config.output_attentions == True


# Load the freefall.txt dataset
with open('datasets/freefall_log.txt', 'r') as f:
    dataset = f.read()

# Preprocess the dataset
# Split the dataset into contexts and questions
# First 3 lines are the context, 4th line is the question, 5th line is the answer

# Split the dataset into contexts and questions
dataset = dataset.split('\n')
contexts = []
questions = []
answers = []

for i in range(0, len(dataset) - 5, 5):
    contexts.append(dataset[i:i + 3])
    questions.append(dataset[i + 3])
    answers.append(dataset[i + 4])

# Convert it to a pandas dataframe
df = pd.DataFrame({'context': contexts, 'question': questions, 'answer': answers})
# Split the dataset into training and validation sets
# 80% training, 20% validation

train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index).reset_index(drop=True)

# Preprocess the contexts and questions
# print(contexts)
# print(questions)
# print(answers)

# Tokenize the contexts and questions
contexts_tokenized = []
questions_tokenized = []
answers_tokenized = []

# Question is always the same so, we can just tokenize it once
question_tokenized = tokenizer(questions[0], **tokenizer_params)

# Tokenize the contexts and answers
for i in range(len(contexts)):
    context_tokenized = tokenizer(' '.join(contexts[i]), **tokenizer_params)
    contexts_tokenized.append(context_tokenized)

    answer_tokenized = tokenizer(answers[i], **tokenizer_params)
    answers_tokenized.append(answer_tokenized)

    questions_tokenized.append(question_tokenized)

# print(contexts_tokenized)
# print(questions_tokenized)
# print(answers_tokenized)

# Convert the tokenized contexts and questions to tensors
contexts_tokenized_tensors = []
questions_tokenized_tensors = []
answers_tokenized_tensors = []

for i in range(len(contexts_tokenized)):
    context_tokenized_tensor = {k: torch.tensor(v[i]) for k, v in contexts_tokenized[i].items()}
    contexts_tokenized_tensors.append(context_tokenized_tensor)

    answer_tokenized_tensor = {k: torch.tensor(v[i]) for k, v in answers_tokenized[i].items()}
    answers_tokenized_tensors.append(answer_tokenized_tensor)

    question_tokenized_tensor = {k: torch.tensor(v) for k, v in questions_tokenized[i].items()}
    questions_tokenized_tensors.append(question_tokenized_tensor)

# print(contexts_tokenized_tensors)

# print(questions_tokenized_tensors)

# print(answers_tokenized_tensors)

# Train the model

# Define the optimizer and the learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

# Define the loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Define the number of epochs
epochs = 1

# Train the model
for epoch in range(epochs):
    print(f'Epoch #{epoch + 1}')
    for i in range(len(contexts_tokenized_tensors)):
        # Zero out the gradients
        optimizer.zero_grad()

        # Get the model outputs
        outputs = model(**contexts_tokenized_tensors[i], **questions_tokenized_tensors[i])

        # Get the loss
        loss = loss_fn(outputs['start_logits'], answers_tokenized_tensors[i]['input_ids'][1:-1]) + loss_fn(
            outputs['end_logits'], answers_tokenized_tensors[i]['input_ids'][1:-1])

        # Back propagate the loss
        loss.backward()

        # Update the parameters
        optimizer.step()

# Evaluate the model
# Define the pipeline
qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)


# Define the function to get the answer
def get_answer(question, context):
    ans = qa_pipeline(question=question, context=context)
    return ans['answer']


# Get the answers
answers = []
for i in range(len(val_df)):
    answer = get_answer(val_df['question'][i], ' '.join(val_df['context'][i]))
    answers.append(answer)

# Print the answers
print(answers)

#Not working. WHY???
# Save the model and tokenizer
