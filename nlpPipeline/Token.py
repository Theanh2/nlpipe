#TOKENIZER PIPELINE

#Train a tokenizer
SUPPORTED_PRETOKENIZER = {"Bytelevel", "BertTokenizer","CharDelimiterSplit", "Whitespace"}
SUPPORTED_TOKENIZER = {"Unigram", "Wordlevel", "WordPiece","BPE" }
SUPPORTED_POSTPROCESSOR = {"BertProcessing", "ByteLevel", "RobertaProcessor"}
SUPPORTED_TRAINERS = {"BPEtrainer", "UnigramTrainer", "WordLevelTrainer", "WordPieceTrainer"}

test_list = [
    "Beautiful is better than ugly."
    "Explicit is better than implicit."
    "Simple is better than complex."
    "Complex is better than complicated."
    "Flat is better than nested."
    "Sparse is better than dense."
    "Readability counts."
]

#1. Initialize Tokenizer
#2. Initialize Normalizer
#3. Initialize Pre Tokenizer
#4. Create List with steps
#5. Run Functions

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers
from tokenizers.normalizers import Lowercase, NFC, NFD, NFKC, NFKD, Nmt, StripAccents
from tokenizers.models import WordPiece,

#-------------------------------------------------------------------------------------------------------------------------

def Init_norm(normalise_list):
    """
    https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#module-tokenizers.normalizers
    :param normalise_list: list
    Supported Normalizers:
    BertNormalizer
    Lowercase
    NFC
    NFD
    NFKC
    NFKD
    Nmt
    StripAccents
    Strip
    :return tokenizers.normalizers.Sequence object
    """
    norm = normalizers.Sequence(normalise_list)
    return norm

#-------------------------------------------------------------------------------------------------------------------------

def Init_pre_token(pre_token):
    """
    https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#module-tokenizers.pre_tokenizers
    """
    if pre_token in SUPPORTED_PRETOKENIZER:
        pre_tokenizer = pre_token
    else:
        print(pre_token + "not defined/supported")
    return(pre_tokenizer)

#-------------------------------------------------------------------------------------------------------------------------

def Init_token(token):
    """
    https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#module-tokenizers.models
    :param token: 
    :return: 
    """
    if token in SUPPORTED_TOKENIZER:
        tokenizer = token
    else:
        print("not supported")
    return(tokenizer)

def Init_post_token():
    bert_tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", 1),
            ("[SEP]", 2),
        ],
    )
#-------------------------------------------------------------------------------------------------------------------------

def token_nlpipe(
        data,
        special_tokens = None,
        tokenizer = None,
        trainer = None,
        pre_tokenizer = None,
        Normalizer = None
):
    """
    Using tokenizers to train tokenizers from scratch
    """
    norm = Init_norm(Normalizer)
    pre_token = pre_tokenizer

    #Any python Iterator
    if hasattr(data,'__iter__') == True:
        tokenizer.train_from_iterator(data, trainer= trainer)

    #Datasets class

    #gzip



