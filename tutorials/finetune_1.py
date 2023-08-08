from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering, Trainer
import torch
import torch.nn.functional as F
import os

# Steps for fine-tuning a model
# 1. Load the model and tokenizer
# 2. Load the dataset
# 3. Preprocess the dataset
# 4. Train the model
# 5. Evaluate the model
# 6. Save the model and tokenizer

# Want to finetune a pre-trained model for question answering

if os.path.exists('models/question_answering/Roberta/'):
    print('Loading model from models/question_answering/Roberta')
    model = AutoModelForQuestionAnswering.from_pretrained('models/question_answering/Roberta')
    tokenizer = AutoTokenizer.from_pretrained('models/question_answering/Roberta')

else:
    print('Downloading model')
    model_name = 'deepset/roberta-base-squad2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

question_answerer = pipeline('question-answering', model=model, tokenizer=tokenizer)

# Load the freefall.txt dataset
with open('data/freefall_log.txt', 'r') as f:
    dataset = f.read()

# Preprocess the dataset
# Split the dataset into contexts and questions
# First 3 lines are the context, 4th line is the question, 5th line is the answer

# Split the dataset into contexts and questions
dataset = dataset.split('\n')
contexts = []
questions = []
answers = []

for i in range(0, len(dataset)-5, 5):
    contexts.append(dataset[i:i+3])
    questions.append(dataset[i+3])
    answers.append(dataset[i+4])

# Preprocess the contexts and questions
# print(contexts)
# print(questions)
# print(answers)

# Tokenize the contexts and questions
contexts_tokenized = []
questions_tokenized = []

# Question is always the same so we can just tokenize it once
question_tokenized = tokenizer(questions[0], return_tensors='pt')


for context in contexts:
    # print(' '.join(context))
    context_tokenized = tokenizer(' '.join(context), return_tensors='pt')
    contexts_tokenized.append(context_tokenized)
    questions_tokenized.append(question_tokenized)

# print(contexts_tokenized)
# print(questions_tokenized)

# Tokenize the answers
answers_tokenized = []
for answer in answers:
    answer_tokenized = tokenizer(answer, return_tensors='pt')
    answers_tokenized.append(answer_tokenized)

# print(answer_tokenized)

# # Train the model using pytorch
# # Use the Adam optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
#
# # Use the CrossEntropyLoss function
# loss_fn = torch.nn.CrossEntropyLoss()
#
# # Train the model
# for epoch in range(3):
#     for i in range(len(contexts_tokenized)):
#         # Get the model outputs
#         print(type(contexts_tokenized[i]))
#         print(type(questions_tokenized[i]))
#         # unpack batch encoding
#
#         outputs = model(**contexts_tokenized[i], **questions_tokenized[i])
#
#
#         # Get the loss
#         loss = loss_fn(outputs.start_logits, torch.tensor([answers_tokenized[i]['input_ids'][0][0]])) + \
#                loss_fn(outputs.end_logits, torch.tensor([answers_tokenized[i]['input_ids'][0][-1]]))
#
#         # Backpropagate the loss
#         loss.backward()
#
#         # Update the parameters
#         optimizer.step()
#
#         # Zero the gradients
#         optimizer.zero_grad()
#
#     print(f'Epoch: {epoch}, Loss: {loss.item()}')
#
# # Evaluate the model
# # Get the predictions
# predictions = []
# for i in range(len(contexts_tokenized)):
#     outputs = model(**contexts_tokenized[i], **questions_tokenized[i])
#     predictions.append(outputs)
#
# # Get the start and end indices
# start_indices = []
# end_indices = []
# for prediction in predictions:
#     start_indices.append(torch.argmax(prediction.start_logits))
#     end_indices.append(torch.argmax(prediction.end_logits))
#
#
# # Get the predicted answer
# predicted_answers = []
# for i in range(len(contexts_tokenized)):
#     predicted_answers.append(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(contexts_tokenized[i]['input_ids'][0][start_indices[i]:end_indices[i]+1])))
#
# print(predicted_answers)

# Create a trainer object
trainer = Trainer(model=model)

# Train the model
trainer.train()

# Why is this not working?
