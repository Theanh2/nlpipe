from transformers import AutoTokenizer
from datasets import load_dataset
import torch
from transformers import (AutoModel,
        AutoModelForAudioClassification,
        AutoModelForCausalLM,
        AutoModelForCTC,
        AutoModelForImageClassification,
        AutoModelForImageSegmentation,
        AutoModelForMaskedLM,
        AutoModelForObjectDetection,
        AutoModelForQuestionAnswering,
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
        AutoModelForSpeechSeq2Seq,
        AutoModelForTableQuestionAnswering,
        AutoModelForTokenClassification)
from transformers import TrainingArguments
from transformers import Trainer
from datasets import load_metric
import numpy as np


SUPPORTED_TASKS = {
    """
    default model selection from transformers
    https://huggingface.co/transformers/v4.12.5/_modules/transformers/pipelines.html#pipeline
    """ 
    "feature-extraction": {
        "pt": (AutoModel,),
        "default": {"model": {"pt": "distilbert-base-cased", "tf": "distilbert-base-cased"}},
    },
    "text-classification": {
        "pt": (AutoModelForSequenceClassification,),
        "default": {
            "model": {
                "pt": "distilbert-base-uncased-finetuned-sst-2-english",
            },
        },
    },
    "token-classification": {
        "pt": (AutoModelForTokenClassification,),
        "default": {
            "model": {
                "pt": "dbmdz/bert-large-cased-finetuned-conll03-english",
            },
        },
    },
    "question-answering": {
        "pt": (AutoModelForQuestionAnswering,),
        "default": {
            "model": {"pt": "distilbert-base-cased-distilled-squad"},
        },
    },
    "table-question-answering": {
        "pt": (AutoModelForTableQuestionAnswering,),
        "default": {
            "model": {
                "pt": "google/tapas-base-finetuned-wtq",
                "tokenizer": "google/tapas-base-finetuned-wtq",
            },
        },
    },
    "fill-mask": {
        "pt": (AutoModelForMaskedLM,),
        "default": {"model": {"pt": "distilroberta-base"}},
    },
    "summarization": {
        "pt": (AutoModelForSeq2SeqLM,),
        "default": {"model": {"pt": "sshleifer/distilbart-cnn-12-6"}},
    },
    "text2text-generation": {
        "pt": (AutoModelForSeq2SeqLM,),
        "default": {"model": {"pt": "t5-base"}},
    },
    "text-generation": {
        "pt": (AutoModelForCausalLM,),
        "default": {"model": {"pt": "gpt2"}},
    },
    "zero-shot-classification": {
        "pt": (AutoModelForSequenceClassification,),
        "default": {
            "model": {"pt": "facebook/bart-large-mnli"},
            "config": {"pt": "facebook/bart-large-mnli"},
            "tokenizer": {"pt": "facebook/bart-large-mnli"},
        },
    },

}
def check(task):
    if task in SUPPORTED_TASKS:
        task = SUPPORTED_TASKS[task]
        targeted_task = task["pt"]
        task_options = task["default"]
    return task

print(check("text-generation"))

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

class fine_nlpipe:

    train_data = None
    test_data = None

    def __init__(self, tokenizer = None,data = None, model = None, extractor = None):
        self.tokenizer = tokenizer
        self.model = model
        self.data = data
        self.training_args = None

    def test(self):
        targeted_task, task_options = check_task(task)
        task_class = targeted_task["impl"]

    def load_data(self, *args, **kwargs):
        """
        see official load_dataset() documentation for more information
        https://huggingface.co/docs/datasets/loading.html
        Takes Local files (csv, json, txt, parquet or python dicts) or any dataset from the huggingface hub
        """
        self.data = load_dataset(*args, **kwargs)

    def split_data(self):
        self.train_data = self.data["train"].select(range(100))
        self.test_data = self.data["test"].select(range(100))

    def set_tokenizer(self, model):
        """
        Set Tokenizer from file (tokenizers package) or uses Autotokenizer with a model from https://huggingface.co/models
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def run_token(self,column, **kwargs):
        self.data = self.data.map(lambda examples: self.tokenizer(examples[column],**kwargs))

    def set_model(self, model = None):
        self.model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

    def set_training_args(self,args):
        self.training_args = TrainingArguments(args)

    def start_tuning(self, **kwargs):
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_data,
            eval_dataset=self.test_data,
            **kwargs
        )
        trainer.train()





# x = fine_nlpipe()
# x.load_data("imdb")
# x.set_tokenizer('bert-base-uncased')
# x.run_token("text", padding="max_length", truncation=True)
# x.split_data()
# x.set_model()

