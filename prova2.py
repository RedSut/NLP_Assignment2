
import nltk
from transformers import pipeline
from transformers import BartForConditionalGeneration, BartTokenizer

model_name = 'facebook/bart-large-cnn'
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

# Function to summarize a slice of the document
def summarize_slice(slice_document):

    model_name = 'sshleifer/distilbart-cnn-12-6'
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)

    
    # Load the BART summarization pipeline
    summarizer = pipeline(task="summarization", model=model, tokenizer=tokenizer)

    # Define the text to be summarized
    input_text = slice_document

    # Generate the summary
    summary = summarizer(input_text, min_length=1, do_sample=False)
    
    #input_text = f"summarize this text: {slice_document}"
    #input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    #outputs = model.generate(input_ids, max_new_tokens=150)

    #return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Return the generated summary 
    return summary[0]['summary_text']

# Function to collate summaries
def collate_summaries(summaries):
    return " ".join(summaries)

def summarize_doc(doc, length):
    summaries = []

    # generate summaries of the slices
    start_index = 0
    while start_index < length:
        slice_doc = slice_document(doc[start_index:], 300) # step size 300 tokens
        summary = summarize_slice(slice_doc)
        summaries.append(summary)
        start_index += 300

    # Collate summaries
    collated_summaries = collate_summaries(summaries)
    return collated_summaries

if __name__ == "__main__":

    with open("input_text.txt", "r", encoding="utf-8") as f:
        input_text = f.read()

    with open("style_text.txt", "r", encoding="utf-8") as f:
        style_text = f.read()


    # Documents
    doc1 = style_text
    doc2 = input_text

    # Constants
    context_window = 800  # Context window limit

    # Measure lengths
    length_doc1 = measure_length(doc1)
    length_doc2 = measure_length(doc2)

    # Compute target lengths 
    target_length_doc1, target_length_doc2 = compute_target_lengths(length_doc1, length_doc2, context_window)

    # Summary document 2
    summary_doc2 = doc2
    while measure_length(summary_doc2) > target_length_doc2:
        summary_doc2 = summarize_doc(summary_doc2, length_doc2) # step of 300 tokens

    # Summary document 1
    summary_doc1 = doc1
    while measure_length(summary_doc1) > target_length_doc1:
        summary_doc1 = summarize_doc(summary_doc1, length_doc1) # step of 300 tokens

    # Save summaries
    with open("summary_document_2.txt", 'w', encoding="utf-8") as f:
        f.write(str(summary_doc2))

    with open("summary_document_1.txt", 'w', encoding="utf-8") as f:
        f.write(str(summary_doc1))

    prompt = f"Summarize the following text: '{summary_doc2}'. Use a style similar to this text: '{summary_doc1}'"

    # Tokenize the prompt
    inputs = tokenizer.encode(prompt, return_tensors='pt', max_length=1024, truncation=True)

    # Generate the summary
    summary_ids = model.generate(inputs, max_new_tokens=800, min_length=1, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Print the summary
    print("The final summary is:")
    print(summary)

    # Save the final summary
    with open("final_summary.txt", 'w', encoding="utf-8") as f:
        f.write(str(summary))