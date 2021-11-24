#TOKENIZER PIPELINE

#Train a tokenizer
test_list = [
    "Beautiful is better than ugly."
    "Explicit is better than implicit."
    "Simple is better than complex."
    "Complex is better than complicated."
    "Flat is better than nested."
    "Sparse is better than dense."
    "Readability counts."
]
import os
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers
from tokenizers.normalizers import Lowercase, NFC, NFD, NFKC, NFKD, Nmt, StripAccents
from tokenizers.models import WordPiece, BPE, Unigram, WordLevel
from tokenizers.pre_tokenizers import Whitespace, ByteLevel, BertPreTokenizer, CharDelimiterSplit
from tokenizers.processors import BertProcessing, ByteLevel, RobertaProcessing
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordLevelTrainer, WordLevelTrainer

SUPPORTED_PRETOKENIZER = {"ByteLevel": ByteLevel,
                          "BertPreTokenizer": BertPreTokenizer,
                          "CharDelimiterSplit": CharDelimiterSplit,
                          "Whitespace": Whitespace}
SUPPORTED_TOKENIZER = {"Unigram": Unigram,
                       "WordPiece": WordPiece,
                       "BPE": BPE,
                       "WordLevel": WordLevel
                       }
SUPPORTED_POSTPROCESSOR = {"BertProcessing": BertProcessing,
                           "ByteLevel": ByteLevel,
                          "RobertaProcessing": RobertaProcessing
                           }
SUPPORTED_TRAINERS = {"BpeTrainer": BpeTrainer,
                      "UnigramTrainer": UnigramTrainer,
                      "WordLevelTrainer": WordLevelTrainer,
                      "WordPieceTrainer": WordLevelTrainer
                      }
#-------------------------------------------------------------------------------------------------------------------------

def Init_norm(tokenizer, normalise_list):
    """
    See official documentation on:
    https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#module-tokenizers.normalizers

    :param tokenizer: <class 'tokenizers.Tokenizer'>
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

    for example: [NFD(), Lowercase(), StripAccents()]
    Uses the normalization techniques in the given order

    :return tokenizers.normalizers.Sequence object
    """

    tokenizer.normalizer = normalizers.Sequence(normalise_list)
    return tokenizer

#-------------------------------------------------------------------------------------------------------------------------

def Init_pre_token(tokenizer, pre_token):
    """
    See official documentation on:
    https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#module-tokenizers.pre_tokenizers

    :param tokenizer: <class 'tokenizers.Tokenizer'>
        Adds pre_tokenizer to existing tokenizer
    :param pre_token: class str
        "ByteLevel",
        "BertPreTokenizer",
        "CharDelimiterSplit",
        "Whitespace"

    :return <class 'tokenizers.Tokenizer'>
    """

    if pre_token in SUPPORTED_PRETOKENIZER:
        tokenizer.pre_tokenizer = SUPPORTED_PRETOKENIZER[pre_token]()
    else:
        print(pre_token + "not supported")
    return tokenizer
#-------------------------------------------------------------------------------------------------------------------------

def Init_token(token, **kwargs):
    """
    See official documentation on:
    https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#module-tokenizers.models

    :param token: str
        "Unigram",
        "WordPiece",
        "BPE",
        "WordLevel"

    :return: <class 'tokenizers.Tokenizer'>
    """

    tokenizer = None
    if token in SUPPORTED_TOKENIZER:
        tokenizer = Tokenizer(SUPPORTED_TOKENIZER[token](**kwargs))
    else:
        print("not supported")
    return(tokenizer)

#-------------------------------------------------------------------------------------------------------------------------

def Init_post_token(tokenizer, post):
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", 1),
            ("[SEP]", 2),
        ],
    )
#-------------------------------------------------------------------------------------------------------------------------

def Init_trainer(trainer, **kwargs):
    """
    See official documentation on:
    https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#module-tokenizers.trainers

    :param trainer: 
        "BpeTrainer",
        "UnigramTrainer", 
        "WordLevelTrainer", 
        "WordPieceTrainer"
    :param kwargs:
        vocab_size
        special_tokens
        show_progress
        ... see more in doc for each trainer
    :return: <class 'tokenizers.trainers'>
    """

    if trainer in SUPPORTED_TRAINERS:
        trainer = SUPPORTED_TRAINERS[trainer](**kwargs)
    else:
        print("not supported")
    return(trainer)

#-------------------------------------------------------------------------------------------------------------------------
def token_nlpipe(
        file_path,
        tokenizer,
        trainer = None,
        pre_tokenizer = None,
        post_tokenizer = None,
        Normalizer = None,
        save = False,
        path = None,
        **kwargs
):
    """
    Wrapper function for training tokenization with tokenizers package
    See official documentation on:
    https://huggingface.co/docs/tokenizers/python/latest/

    :param
    :param
    :param
    :param
    :param
    :param
    :return saves tokenizer in specified path, returns tokenizer
    """
    files = [file_path]
    cpath = os.getcwd()

    tnz = Init_token(tokenizer, **kwargs)
    if Normalizer is not None:
        tnz = Init_norm(tnz, Normalizer)
    if pre_tokenizer is not None:
        tnz = Init_pre_token(tnz, pre_tokenizer)
    if post_tokenizer is not None:
        tnz = Init_post_token(tnz, post)

    train = Init_trainer(trainer,**kwargs)

    tnz.train(files, train)

    if save is True:
        if path is None:
            cpath = cpath.replace(os.sep, '/')
            tnz.save(cpath + "/trained_tokenizer.json")
            tokenizer = Tokenizer.from_file(cpath + "/trained_tokenizer.json")
        else:
            tnz.save(path + "/trained_tokenizer.json")
            tokenizer = Tokenizer.from_file(path + "/trained_tokenizer.json")
    return tokenizer


# -------------------------------------------------------------------------------------------------------------------------



#TESTING
# tokenizer = Init_token("BPE",unk_token = "[UNK]")
# tokenizer = Init_norm(tokenizer, [Lowercase()])
# tokenizer = Init_pre_token(tokenizer, "Whitespace")
# train = Init_trainer("BpeTrainer",special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
# path = "C:/Users/thean/Documents/tests/wikitext-103-raw/wiki.test.raw"
# files = [path]
# tokenizer.train(files, train)
# #tokenizer.train_from_iterator(files, train)
# tokenizer.save("C:/Users/thean/Documents/tests/wikitext-103-raw/tokenizer-wiki.json")
#
# tokenizer = Tokenizer.from_file("C:/Users/thean/Documents/tests/wikitext-103-raw/tokenizer-wiki.json")
# output = tokenizer.encode("Simple is How are you 😁 ?")
# print(output.tokens)

token_nlpipe(file_path = "C:/Users/thean/Documents/tests/wikitext-103-raw/wiki.test.raw",
             tokenizer = "BPE",
             unk_token = "[UNK]",
             trainer = "BpeTrainer",
             special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
             pre_tokenizer = "Whitespace",
             Normalizer = [Lowercase()],
             save = True,
             path = "C:/Users/thean/Documents/tests/wikitext-103-raw/"
)

tokenizer = Tokenizer.from_file("C:/Users/thean/Documents/tests/wikitext-103-raw/trained_tokenizer.json")
output = tokenizer.encode("Simple is How are you 😁 ?")
print(output.tokens)