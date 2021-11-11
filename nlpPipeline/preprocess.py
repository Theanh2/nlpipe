#Part of Speech Tag
def POS(input):
    """Parts of Speech Tag with nltk

    """
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize

    tokenized = sent_tokenize(input)
    for i in tokenized:
        wordsList = nltk.word_tokenize(i)
        tagged = nltk.pos_tag(wordsList)
        return tagged

#Lemmatizer
def lemmatize(input):
    """Wordnet Lemmatizer with nltk
    Needs words with appropriate POS tags
    """
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(input)

# Stemmer
def stemming(input):
    """stemming with Porterstemmer

    """
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()

    for w in input:
        return ps.stem(w)

# #Stop Word Removal
def removestopwords(input):
    """removing stopwords

    """
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    stop_words = set(stopwords.words('english'))

    word_tokens = word_tokenize(input)

    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    filtered_sentence = []

    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    return(filtered_sentence)

# #Normalization
def normalize(input):
    """Normalization described by Emma Flint, Elliot Ford, Olivia Thomas, Andrew Caines & Paula Buttery (2016) - A Text Normalisation System for Non-Standard Words.
    https://github.com/EFord36/normalise
    """
    from normalise import normalise
    normalise(input, verbose=True)

# #Named Entity Recognition
# def ner():
#
# #Parsing
# def parse():
#

#remove punctuation
def removepunctuation(input):
    """remove punctuation
    potentially faster with regex
    """
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for w in input:
        if w in punc:
            input = input.replace(w, "")

    return input

#remove special characters
# def removespecial():


#lowercase
def lowercase(input):
    """lowercasing whole string

    """
    return input.lower()
