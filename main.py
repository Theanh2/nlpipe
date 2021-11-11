import pandas as pd
from IPython.display import display

data = pd.read_csv('nlpPipeline/WikiQA-dev.txt', delimiter = "\t", header=None)
data.columns = ["question", "answer", "true"]
data.head()
display(data)