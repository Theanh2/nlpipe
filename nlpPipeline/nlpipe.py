test_list = [
    "Beautiful is better than ugly."
    "Explicit is better than implicit."
    "Simple is better than complex."
    "Complex is better than complicated."
    "Flat is better than nested."
    "Sparse is better than dense."
    "Readability counts."
]




from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from datasets import load_dataset
from tokenizers import Tokenizer
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer, LancasterStemmer, Cistem, RSLPStemmer, SnowballStemmer, WordNetLemmatizer

SUPPORTED_STEMMER = {
        "Cistem": Cistem,
        "LancasterStemmer": LancasterStemmer,
        "PorterStemmer": PorterStemmer,
        "RSLPStemmer":  RSLPStemmer,
    "SnowballStemmer": SnowballStemmer
}
SUPPORTED_LEMMA = {"WordNet": WordNetLemmatizer}

SUPPORTED_VECTORIZER = {
    "BOW": CountVectorizer,
    "TFIDF": TfidfVectorizer

}


#-------------------------------------------------------------------------------------------------------------------------
#Count Vectorizer (BOW)

def Countvec(data):
    vectorizer = CountVectorizer()
    out = vectorizer.fit_transform(data)
    return out
#print(Countvec(test_list))

#-------------------------------------------------------------------------------------------------------------------------
#tf-idf vectorizer
def TFIDF(data):
    vectorizer = TfidfVectorizer()
    out = vectorizer.fit_transform(data)
    return out
#print(TFIDF(test_list))

#-------------------------------------------------------------------------------------------------------------------------
#stemming mapping
def processing(example, column,stoplist = None, stemmer = None, lemmatizer = None, languageSnowball = 'english'):
    """
    Pre-processing Pipeline with stopword removal, stemmer and lemmatizer
    :param stoplist: List of stop-words that get removed from the dataset
    :param stemmer: One of the supported Stemmer (nltk)
            Cistem()
            LancasterStemmer()
            PorterStemmer()
            RSLPStemmer()
            SnowballStemmer()
    :param lemmatizer: One of the supported Lemmatizer(nltk)
        WordNetLemmatizer
    """
    stem_list  = word_tokenize(example[column])

    if stoplist is not None:
        stop_list = stoplist
        stem_list = [word for word in stem_list if not word in stop_list]
    if stemmer is not None:
        if stemmer == "SnowballStemmer":
            stemmer = SUPPORTED_STEMMER[stemmer](languageSnowball)
            stem_list = [stemmer.stem(word) for word in stem_list]
        else:
            stemmer = SUPPORTED_STEMMER[stemmer]()
            stem_list = [stemmer.stem(word) for word in stem_list]
    if lemmatizer is not None:
        lemmatizer = SUPPORTED_LEMMA[lemmatizer]()
        stem_list = [lemmatizer.lemmatize(word) for word in stem_list]

    stem_list = ' '.join(stem_list)
    example['sentence1'] = stem_list
    return example

#-------------------------------------------------------------------------------------------------------------------------
#random forest

#-------------------------------------------------------------------------------------------------------------------------
#logistics regression

#-------------------------------------------------------------------------------------------------------------------------
#word2vec,CBOW, skipgram

#-------------------------------------------------------------------------------------------------------------------------
#Naive Bayes

#-------------------------------------------------------------------------------------------------------------------------
#SVM
#-------------------------------------------------------------------------------------------------------------------------

tokenizer = Tokenizer.from_file("C:/Users/thean/Documents/tests/wikitext-103-raw/trained_tokenizer.json")
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
#
#dataset = load_dataset('glue', 'mrpc', split='train')
class nlpipe:

    processed_data = None
    extracted = None
    def __init__(self, tokenizer = None,data = None, model = None):
        self.tokenizer = tokenizer
        self.model = model
        self.data = data

    def load_data(self, *args, **kwargs):
        """
        see official load_dataset() documentation for more information
        https://huggingface.co/docs/datasets/loading.html
        Takes Local files (csv, json, txt, parquet or python dicts) or any dataset from the huggingface hub
        """
        self.data = load_dataset(*args, **kwargs)


    def run_token(self,column, batched = False):
        self.data = self.data.map(lambda examples: self.tokenizer(examples[column]),
                                  batched=batched)

    def run_processing(self, column,stoplist = None, stemmer = None, lemmatizer = None):
        self.data = self.data.map(lambda example: processing(example,
                                                             column = column,
                                                             stoplist = stoplist,
                                                             stemmer = stemmer,
                                                             lemmatizer = lemmatizer,
                                                             languageSnowball = 'english'
                                                            )
        )

    def run_extractor(self, column):
        self.extracted = self.tokenizer.fit_transform(self.data[column])
        return self.extracted

    #change after initializing
    def set_seq(self, seq):
        """
        stop word removal
        stemming
        lemmatization
        """
        self.pre_processing_seq = seq

    def set_model(self, model):
        """
        supervised:
        Logistic Regression model
        Naive Bayes
        Random Forest
        SVMs

        unsupervised:
        BERT
        RoBERTa
        ALBERT

        """
        self.model = model

    def set_tokenizer(self,Vectorizer = None ,Transformer = None, from_file = None):
        """
        Set Tokenizer from file (tokenizers package) or uses Autotokenizer with a model from https://huggingface.co/models
        """
        if Vectorizer is not None:
            self.tokenizer = SUPPORTED_VECTORIZER[Vectorizer]()

        if from_file is not None:
            tokenizer = Tokenizer.from_file(from_file)
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

        if Transformer is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(Transformer)


    def run_model_pipe(self, model=None, metric=None):
        """
        :param data from datasets class
        """
        #Run preprocessing

        #run model
        #cross validation
        #random seed
#amrozi accus hi brother , whom he call `` the wit `` , of deliber distort hi evid .

english_sw = ["his", "brother"]

x = nlpipe()
x.load_data('glue', 'mrpc', split='train')
x.set_tokenizer(Vectorizer = "BOW")
print(x.run_extractor('sentence1'))

