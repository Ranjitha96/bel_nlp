# bel_nlp
APPROCH:
We use python and a natural language processing (NLP) technique known as Term Frequency — Inverse Document Frequency (tf-idf) to summarize documents.We'll be using sklearn along with nltk to accomplish this task.

GOAL:
Our goal is to take a given document, wether that’s a blog post, news article, random website, or any other corpus, and extract a few sentences that best summarized the document.

STEPS:
Below is the outline of steps needed:
  1. Preprocess the document.
  2. Import a corpus used for training.
  3. Create a count vector.
  4. Build a tf-idf matrix.
  5. Score each sentence.
  6. Summarize using top ranking sentences.
