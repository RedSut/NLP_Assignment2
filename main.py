import nltk
from transformers import pipeline
from transformers import BartForConditionalGeneration, BartTokenizer
import textwrap


# BART LLM MODEL
model_name = 'sshleifer/distilbart-cnn-12-6'
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)


# Function to measure the length of a document in tokens
def measure_length(document):
    tokens = nltk.word_tokenize(document)
    return len(tokens)


# Function to compute target lengths proportionally
def compute_target_lengths(length_doc1, length_doc2, context_window):
    total_length = length_doc1 + length_doc2
    target_length_doc1 = int(length_doc1 / total_length * context_window)
    target_length_doc2 = int(length_doc2 / total_length * context_window)
    return target_length_doc1, target_length_doc2


# Function to slice the document
def slice_document(document, target_length):
    tokens = nltk.word_tokenize(document)
    return " ".join(tokens[:target_length])
    #return document[:target_length]


# Function to summarize a slice of the document
def summarize_slice(slice_document):
    
    # Load the BART summarization pipeline
    summarizer = pipeline(task="summarization", model=model, tokenizer=tokenizer)

    # Define the text to be summarized
    input_text = slice_document

    # Generate the summary
    summary = summarizer(input_text, max_length=1024, min_length=1, do_sample=False)

    formatted_summary = "\n".join(textwrap.wrap(summary[0]['summary_text'], width=80))

    return formatted_summary


# Function to collate summaries
def collate_summaries(summaries):
    return " ".join(summaries)


def summarize_doc(doc):
    summaries = []

    # Generate summaries of the slices
    start_index = 0
    while start_index < len(doc):
        slice_doc = slice_document(doc[start_index:], 300) # step size 300
        summary = summarize_slice(slice_doc)
        summaries.append(summary)
        start_index += len(slice_doc)

    # Collate summaries
    collated_summaries = collate_summaries(summaries)
    return collated_summaries


if __name__ == "__main__":
    
    # Initialize documents
    with open("inputs/input_text.txt", "r", encoding="utf-8") as f:
        input_text = f.read()

    with open("inputs/style_text.txt", "r", encoding="utf-8") as f:
        style_text = f.read()

    doc1 = style_text
    doc2 = input_text

    # Constants
    context_window = 800  # Context window limit adjusted

    # Measure lengths
    length_doc1 = measure_length(doc1)
    length_doc2 = measure_length(doc2)

    # Compute target lengths 
    target_length_doc1, target_length_doc2 = compute_target_lengths(length_doc1, length_doc2, context_window)

    # Summary document 2
    summary_doc2 = doc2
    while measure_length(summary_doc2) > target_length_doc2:
        summary_doc2 = summarize_doc(summary_doc2)

    # Summary document 1
    summary_doc1 = doc1
    while measure_length(summary_doc1) > target_length_doc1:
        summary_doc1 = summarize_doc(summary_doc1)

    # Save summaries
    with open("summaries/summary_input_text.txt", 'w', encoding="utf-8") as f:
        f.write(str(summary_doc2))

    with open("summaries/summary_style_text.txt", 'w', encoding="utf-8") as f:
        f.write(str(summary_doc1))

    # Prompt for final summarization
    prompt = f"Summarize the following text: '{summary_doc2}'. Use a style similar to this text: '{summary_doc1}'"

    # Tokenize the prompt
    inputs = tokenizer.encode(prompt, return_tensors='pt', max_length=1024, truncation=True)

    # Generate the summary
    summary_ids = model.generate(inputs, max_new_tokens=800, min_length=1, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    formatted_summary = "\n".join(textwrap.wrap(summary, width=80))

    # Print the summary
    print("The final summary is:")
    print(formatted_summary)

    # Save the final summary
    with open("final_summary.txt", 'w', encoding="utf-8") as f:
        f.write(str(formatted_summary))