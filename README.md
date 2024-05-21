# NLP Assignment 1
# Author: Davide Sut 
# ID: VR505441

## The Assignment
The requested task is to implement an algorithm to generate summarization of an input text following the style of another text given as input, using a Large Language Model (LLM).

The implementation uses a BART LLM model (_sshleifer/distilbart-cnn-12-6_) for the summarization tasks.

The documents are taken as input using the files inside the _\inputs_ folder, then their lengths are measured (in number of tokens) and are calculated the target lengths in order to check if the documents can fit in the context window of the LLM.

If a document is larger than the target length the following steps are executed:
- A constant step size is defined (in number of tokens);
- The document is sliced from start to the size step;
- The slice is summarized using the LLM without setting a maximum length (max_length = model max_length for security purposes);
- The new start is updated according to the step size;
- The previous three steps are repeated till the end of the document;
- The created summaries are collated together;
- All the previous steps are repeated until the final summary length is within the target length.

Then the summaries (or the plain documents) are saved in two separated files in the _\summaries_ folder.

Finally the last summary is created following the chosen style using the same LLM.

The result is given as a console output and is saved in the file: _final\_summary.txt_.


## Instructions

You simply need to put the text to summarize in the _input\_text.txt_ file and the style text for the summarization in the _style\_text.txt_ file.

Then run the _main.py_ script and the console outputs the final summary.

## Warnings

During the execution of the program some warnings will be shown caused by the definition of the maximum summary length. This length is deliberately equal to the maximum length accepted by the LLM as the assignment explicitly required not to worry about the length of the summaries.