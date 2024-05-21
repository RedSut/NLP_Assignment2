from transformers import pipeline

import nltk
from nltk.tokenize import sent_tokenize

# Function to measure the length of a document in tokens
def measure_length(document):
    tokens = nltk.word_tokenize(document)
    return len(tokens)


input_text = """Jannik Sinner, the rising star of Italian tennis, embodies raw talent and determination. 
Born on August 16, 2001, in Innichen, Italy, Sinner's meteoric rise in the tennis world has been nothing short of remarkable. His aggressive baseline style, coupled with precise shot-making and remarkable footwork, has captivated fans worldwide.
At just 23, Sinner has already made waves on the ATP Tour, securing victories against some of the game's biggest names. His breakthrough moment came in 2019 when he won the Next Gen ATP Finals, showcasing his immense potential. Since then, he has continued to impress, reaching career-high rankings and thrilling spectators with his fearless play.
Off the court, Sinner exudes humility and focus, displaying a maturity beyond his years. He approaches each match with a calm demeanor and a relentless drive to succeed. As he continues to hone his skills and navigate the challenges of professional tennis, one thing remains certain: Jannik Sinner is a force to be reckoned with, poised to leave an indelible mark on the sport for years to come.
"""

with open("input_text.txt", "r", encoding="utf-8") as f:
    new_text = f.read()

print(measure_length(new_text))

# Load the BART summarization pipeline
summarizer = pipeline("summarization")

# Generate the summary
summary = summarizer(new_text, model="sshleifer/distilbart-cnn-12-6", do_sample=False)

# Print the generated summary
print("Generated Summary:")
print(summary[0]['summary_text'])