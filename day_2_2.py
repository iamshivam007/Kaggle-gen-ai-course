import os

import google.generativeai as genai
from google.api_core import retry
from chromadb import Documents, EmbeddingFunction, Embeddings, Client


GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)


## Calculate similarity scores
texts = [
    'The quick brown fox jumps over the lazy dog.',
    'The quick rbown fox jumps over the lazy dog.',
    'teh fast fox jumps over the slow woofer.',
    'a quick brown fox jmps over lazy dog.',
    'brown fox jumping over dog',
    'fox > dog',
    # Alternative pangram for comparison:
    'The five boxing wizards jump quickly.',
    # Unrelated text, also for comparison:
    'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus et hendrerit massa. Sed pulvinar, nisi a lobortis sagittis, neque risus gravida dolor, in porta dui odio vel purus.',
]


response = genai.embed_content(model='models/text-embedding-004',
                               content=texts,
                               task_type='semantic_similarity')



def truncate(t: str, limit: int = 50) -> str:
  """Truncate labels to fit on the chart."""
  if len(t) > limit:
    return t[:limit-3] + '...'
  else:
    return t

truncated_texts = [truncate(t) for t in texts]


import pandas as pd
import seaborn as sns


# Set up the embeddings in a dataframe.
df = pd.DataFrame(response['embedding'], index=truncated_texts)
# Perform the similarity calculation
sim = df @ df.T

"""
A similarity score of two embedding vectors can be obtained by calculating their inner product. If 
 is the first embedding vector, and 
 the second, this is 
. As these embedding vectors are normalised to unit length, this is also the cosine similarity.

This score can be computed across all embeddings through the matrix self-multiplication: df @ df.T.

Note that the range from 0.0 (completely dissimilar) to 1.0 (completely similar) is depicted in the heatmap from dark (0.0) to light (1.0).
"""

# Draw!
sns.heatmap(sim, vmin=0, vmax=1);
