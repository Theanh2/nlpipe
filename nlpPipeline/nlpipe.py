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
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
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
#Processing help function, supports stop word removal, stemmer and lemmatizer from nltk package
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
#logistic regression

#-------------------------------------------------------------------------------------------------------------------------

class nlpipe:

    Naiveclassifier = None
    Logisticclassifier = None
    SVMclassifier = None
    Randomclassifier = None
    extracted = None
    train_data = None
    test_data = None

    def __init__(self, tokenizer = None,data = None, model = None, extractor = None):
        self.tokenizer = tokenizer
        self.model = model
        self.data = data
        self.extractor = extractor

    def load_data(self, *args, **kwargs):
        """
        see official load_dataset() documentation for more information
        https://huggingface.co/docs/datasets/loading.html
        Takes Local files (csv, json, txt, parquet or python dicts) or any dataset from the huggingface hub
        """
        self.data = load_dataset(*args, **kwargs)

    def split_data(self,**kwargs):
        """
        see official train_test_split for more information about parameters
        https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.train_test_split
        """
        temp = self.data.train_test_split(**kwargs)
        self.train_data = temp["train"]
        self.test_data = temp["test"]

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

    def Randomforest(self,feature_column, target_column, extractor, **kwargs):
        """
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        """
        if self.train_data is None:
            return

        self.extractor = SUPPORTED_VECTORIZER[extractor]()

        train_vec = self.extractor.fit_transform(self.train_data[feature_column])
        self.Randomclassifier = RandomForestClassifier(**kwargs).fit(train_vec, self.train_data[target_column])

        test_vec = self.extractor.transform(self.test_data[feature_column])
        predicted = self.Randomclassifier.predict(test_vec)

        print("classification report (Random Forest):")
        print(metrics.classification_report(self.test_data[target_column], predicted))
        print("confusion matrix (Random Forest):")
        print(metrics.confusion_matrix(self.test_data[target_column], predicted))

    def linearSVM(self,feature_column, target_column, extractor, **kwargs):
        """
        linear SVM by default with stochastic gradient descent
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
        optional parameters for SGD training
        """
        if self.train_data is None:
            return

        self.extractor = SUPPORTED_VECTORIZER[extractor]()

        train_vec = self.extractor.fit_transform(self.train_data[feature_column])
        self.SVMclassifier = SGDClassifier(**kwargs).fit(train_vec, self.train_data[target_column])

        test_vec = self.extractor.transform(self.test_data[feature_column])
        predicted = self.SVMclassifier.predict(test_vec)

        print("classification report (SVM):")
        print(metrics.classification_report(self.test_data[target_column], predicted))
        print("confusion matrix (SVM):")
        print(metrics.confusion_matrix(self.test_data[target_column], predicted))


    def NaiveB(self,feature_column, target_column, extractor, **kwargs):
        """
        Parameters for MultinomialNB()
        https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB
        """
        if self.train_data is None:
            return

        self.extractor = SUPPORTED_VECTORIZER[extractor]()

        train_vec = self.extractor.fit_transform(self.train_data[feature_column])
        self.Naiveclassifier = MultinomialNB(**kwargs).fit(train_vec, self.train_data[target_column])

        test_vec = self.extractor.transform(self.test_data[feature_column])
        predicted = self.Naiveclassifier.predict(test_vec)

        print("classification report (Naive Bayes):")
        print(metrics.classification_report(self.test_data[target_column], predicted))
        print("confusion matrix (Naive Bayes):")
        print(metrics.confusion_matrix(self.test_data[target_column], predicted))

    def logistic(self, feature_column, target_column, extractor, **kwargs):
        """
        Parameters for LogisticRegression()
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
        'log' loss for logistic regression
        optional parameters for SGD training
        """
        if self.train_data is None:
            return

        self.extractor = SUPPORTED_VECTORIZER[extractor]()

        train_vec = self.extractor.fit_transform(self.train_data[feature_column])
        self.Logisticclassifier = SGDClassifier(loss = 'log', **kwargs).fit(train_vec, self.train_data[target_column])

        test_vec = self.extractor.transform(self.test_data[feature_column])
        predicted = self.Logisticclassifier.predict(test_vec)

        print("classification report (Logistic Regression):")
        print(metrics.classification_report(self.test_data[target_column], predicted))
        print("confusion matrix (Logistic Regression):")
        print(metrics.confusion_matrix(self.test_data[target_column], predicted))

    def run_extractor(self,data = self.data, extractor, column):
        """
        returns extracted Features
        See official documentation for CountVectorizer and TFIDFvectorizer
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer
        """
        self.extractor = SUPPORTED_VECTORIZER[extractor]()
        self.extracted = self.extractor.fit_transform(data[column])
        return self.extracted


    def set_tokenizer(self,Transformer = None, from_file = None):
        """
        Set Tokenizer from file (tokenizers package) or uses Autotokenizer with a model from https://huggingface.co/models
        """

        if from_file is not None:
            tokenizer = Tokenizer.from_file(from_file)
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

        if Transformer is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(Transformer)



x = nlpipe()
x.load_data('glue', 'mrpc', split='train')
x.split_data(test_size = 0.1, seed = 1221, shuffle = False)
print(x.run_extractor("BOW", 'sentence1'))
# x.logistic('sentence1', 'label', 'BOW')
# x.NaiveB('sentence1', 'label', 'TFIDF')
# x.Randomforest('sentence1', 'label', 'BOW', n_estimators = 1000)
# x.linearSVM('sentence1', 'label', 'BOW',loss='log')


#tokenizer = Tokenizer.from_file("C:/Users/thean/Documents/tests/wikitext-103-raw/trained_tokenizer.json")


