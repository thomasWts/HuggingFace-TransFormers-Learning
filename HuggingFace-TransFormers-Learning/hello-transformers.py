from transformers import pipeline
print(pipeline('sentiment-analysis')("I love using transformers!"))