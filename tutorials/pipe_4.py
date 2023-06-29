from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import torch
import torch.nn.functional as F
import os

# Now let's test it while specifying the model and tokenizer
# If model exists, load it, otherwise download it
if os.path.exists('models/question_answering/'):
    print('Loading model from models/question_answering')
    model = AutoModelForQuestionAnswering.from_pretrained('models/question_answering')
    tokenizer = AutoTokenizer.from_pretrained('models/question_answering')

else:
    print('Downloading model')
    model_name = 'distilbert-base-uncased-distilled-squad'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

question_answerer = pipeline('question-answering', model=model, tokenizer=tokenizer)

context = 'Two objects of the same weight are dropped from different heights. Object 1 is heavier than object 2 ' \
          'but object 2 is dropped from a greater height.'

question = 'Which object will hit the ground first?'

res = question_answerer(question=question, context=context)
print('Question: ', question)
print(f"Answer: '{res['answer']}', score: {round(res['score'], 4)}, start: {res['start']}, end: {res['end']}")


context = 'Object 1 weights 60kg while object 2 weighs 50kg'
question = 'Which object is weighs more? object 1, object 2 or the same?'

res = question_answerer(question=question, context=context)
print('Question: ', question)
print(f"Answer: '{res['answer']}', score: {round(res['score'], 4)}, start: {res['start']}, end: {res['end']}")





# Save the model and tokenizer
model.save_pretrained('models/question_answering')
tokenizer.save_pretrained('models/question_answering')