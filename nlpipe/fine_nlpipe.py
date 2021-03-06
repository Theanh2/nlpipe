from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import (AutoModel,
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
        AutoModelForQuestionAnswering,
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
        AutoModelForTableQuestionAnswering,
        AutoModelForTokenClassification)
from transformers import TrainingArguments
from transformers import Trainer
from datasets import load_metric
import numpy as np

SUPPORTED_TASKS = {"AutoModel":AutoModel.from_pretrained,
                   "AutoModelForSequenceClassification": AutoModelForSequenceClassification.from_pretrained,
                   "AutoModelForCausalLM":AutoModelForCausalLM.from_pretrained,
                   "AutoModelForMaskedLM":AutoModelForMaskedLM.from_pretrained,
                   "AutoModelForQuestionAnswering":AutoModelForQuestionAnswering.from_pretrained,
                   "AutoModelForSeq2SeqLM":AutoModelForSeq2SeqLM.from_pretrained,
                   "AutoModelForTableQuestionAnswering":AutoModelForTableQuestionAnswering.from_pretrained,
                   "AutoModelForTokenClassification":AutoModelForTokenClassification.from_pretrained
                   }

class fine_nlpipe:

    train_data = None
    test_data = None
    trainer = None
    def __init__(self, tokenizer = None,data = None, model = None):
        self.tokenizer = tokenizer
        self.model = model
        self.data = data
        self.training_args = None

    def load_data(self, *args, **kwargs):
        """
        see official load_dataset() documentation for more information
        https://huggingface.co/docs/datasets/loading.html
        Takes Local files (csv, json, txt, parquet or python dicts) or any dataset from the huggingface hub
        """
        self.data = load_dataset(*args, **kwargs)

    def split_data(self,train_size = 1, test_size = 1, seed = 1234 ):
        """
        Splits a dataset object with loading script into train and test split
        """
        train_size = len(self.data["train"])*train_size
        if train_size > len(self.data["train"]):
            train_size = len(self.data["train"])

        test_size = len(self.data["test"])*test_size
        if test_size > len(self.data["test"]):
            test_size = len(self.data["test"])

        self.train_data = self.data["train"].shuffle(seed =seed).select(range(int(train_size)))
        self.test_data = self.data["test"].shuffle(seed =seed).select(range(int(test_size)))


    def set_tokenizer(self,Transformer = None, from_file = None):
        """
        Set Tokenizer from file (tokenizers package) or uses Autotokenizer with a model from https://huggingface.co/models
        """

        if from_file is not None:
            tokenizer = Tokenizer.from_file(from_file)
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

        if Transformer is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(Transformer)

    def run_token(self,column,batched = True, **kwargs):
        print("Mapping batches:")
        self.data = self.data.map(lambda examples: self.tokenizer(examples[column], **kwargs), batched = batched)

    def set_model(self, task, model, **kwargs):
        self.model = SUPPORTED_TASKS[task](model, **kwargs)

    def set_training_args(self,*args, **kwargs):
        self.training_args = TrainingArguments(*args, **kwargs)

    def compute_metrics(self,eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)

    def set_metric(self,*args ,**kwargs):
        self.metric = load_metric(*args, **kwargs)
    
    def start_tuning(self,train_dataset = None,eval_dataset = None,save_path = None, **kwargs):
        if train_dataset is None:
            train_dataset = self.train_data
        if eval_dataset is None:
            eval_dataset = self.test_data

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs
        )
        self.trainer.evaluate()
        if save_path is not None:
            self.trainer.save_model(save_path)



# -------------------------------------------------------------------------------------------------------------------------
#CODE EXAMPLE
# pipe = fine_nlpipe()
# pipe.set_metric("glue", "mrpc")
# pipe.load_data("imdb")
# pipe.set_tokenizer(Transformer = 'bert-base-cased')
# pipe.run_token("text", padding = "max_length", truncation = True)
# pipe.split_data(seed = 42, train_size= 0.001, test_size= 0.001 )
# pipe.set_model("AutoModelForSequenceClassification", "bert-base-cased", num_labels = 2)
# pipe.set_training_args("test_trainer", evaluation_strategy="epoch")
# pipe.start_tuning(save_path = "C:/", compute_metrics = pipe.compute_metrics)


