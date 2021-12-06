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
from tokenizers.processors import BertProcessing, ByteLevel, RobertaProcessing, TemplateProcessing
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordPieceTrainer, WordLevelTrainer

SUPPORTED_PRETOKENIZER = {"ByteLevel": ByteLevel,
                          "BertPreTokenizer": BertPreTokenizer,
                          "CharDelimiterSplit": CharDelimiterSplit,
                          "Whitespace": Whitespace}

SUPPORTED_POSTPROCESSOR = {"BertProcessing": BertProcessing,
                           "ByteLevel": ByteLevel,
                          "RobertaProcessing": RobertaProcessing,
                        "TemplateProcessing": TemplateProcessing
                           }
SUPPORTED_TRAINERS = {"BpeTrainer": BpeTrainer,
                      "UnigramTrainer": UnigramTrainer,
                      "WordLevelTrainer": WordLevelTrainer,
                      "WordPieceTrainer": WordPieceTrainer
                      }
#Tokenizer model defaults from transformers model defaults
SUPPORTED_TOKENIZER = {"Unigram": {"model": Unigram,
                                   "default": {"token_param": {"vocab":None},
                                       "Normalizer": [NFD(), Lowercase(), StripAccents()],
                                                 "pre_tokenizer":  "Whitespace",
                                                 "post": "BertProcessing",
                                                 "trainer": "UnigramTrainer",
                                                 "train_param": {"unk_token": "[UNK]",
                                                                "special_tokens": ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]},
                                                 "post_param": {"sep": ("[SEP]",2),"cls":("[CLS]",1)}
                                                 }
                                   },
                       "WordPiece": {"model": WordPiece,
                                     #default is for BERT
                                     "default": {"token_param": {"unk_token": "[UNK]"},
                                                 "Normalizer": [NFD(), Lowercase(), StripAccents()],
                                                 "pre_tokenizer":  "Whitespace",
                                                 "post": "BertProcessing",
                                                 "trainer": "WordPieceTrainer",
                                                 "train_param": {"vocab_size": 30522,
                                                                "special_tokens": ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]},
                                                 "post_param": {"sep": ("[SEP]",2),"cls":("[CLS]",1)}
                                                 }
                                   },
                       "BPE": {"model": BPE,
                               #default is for RoBERTa
                              "default": {       "token_param": {"unk_token": "<unk>"},
                                                 "Normalizer": [NFD(), Lowercase(), StripAccents()],
                                                 "pre_tokenizer":  "Whitespace",
                                                 "post": "RobertaProcessing",
                                                 "trainer": "BpeTrainer",
                                                 "train_param": {"vocab_size": 30522,
                                                                "special_tokens": [ "<s>", "<pad>", "</s>","<unk>", "<mask>"]},
                                                 "post_param": {"sep": ("[</s>]",2),"cls":("[<s>]",1)}
                                                 }
                                   },
                       "WordLevel": {"model": WordLevel,
                                     "default": {"token_param": {"unk_token": "[UNK]"},
                                                 "Normalizer": [NFD(), Lowercase(), StripAccents()],
                                                 "pre_tokenizer":  "Whitespace",
                                                 "post": "BertProcessing",
                                                 "trainer": "WordLevelTrainer",
                                                 "train_param": {"special_tokens": ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]},
                                                 "post_param": {"sep": ("[SEP]",2),"cls":("[CLS]",1)}
                                                 }
                                   }
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
        tokenizer = Tokenizer(SUPPORTED_TOKENIZER[token]["model"](**kwargs))
    else:
        print("not supported")
    return(tokenizer)

#-------------------------------------------------------------------------------------------------------------------------

def Init_post_token(tokenizer,post, **kwargs):

    tokenizer.post_processor = SUPPORTED_POSTPROCESSOR[post](**kwargs)
    return (tokenizer)

        # single="[CLS] $A [SEP]",
        # pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        # special_tokens=[
        #     ("[CLS]", 1),
        #     ("[SEP]", 2),
        # ]
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
        post = None,
        Normalizer = None,
        save = False,
        path = None,
        token_param = None,
        train_param = None,
        post_param = None,
        default = False,
        file_name = "trained_tokenizer",
        add_post = False
):
    """
    Wrapper function for training tokenization with tokenizers package
    See official documentation on:
    https://huggingface.co/docs/tokenizers/python/latest/
    """

    files = file_path
    cpath = os.getcwd()

    if default == True:
        default = SUPPORTED_TOKENIZER[tokenizer]["default"]

        if "token_param" in default.keys():
            if token_param is None:
                token_param = default["token_param"]
        if "trainer" in default.keys():
            if trainer is None:
                trainer = default["trainer"]
        if "pre_tokenizer" in default.keys():
            if pre_tokenizer is None:
                pre_tokenizer = default["pre_tokenizer"]
        if "Normalizer" in default.keys():
            if Normalizer is None:
                Normalizer = default["Normalizer"]
        if "train_param" in default.keys():
            if train_param is None:
                train_param = default["train_param"]
        if "post_param" in default.keys():
            if post_param is None:
                post_param = default["post_param"]
        if "post" in default.keys():
            if post is None:
                post = default["post"]


    tnz = Init_token(tokenizer, **token_param)
    if Normalizer is not None:
        tnz = Init_norm(tnz, Normalizer)
    if pre_tokenizer is not None:
        tnz = Init_pre_token(tnz, pre_tokenizer)

    if post is not None:
        tnz = Init_post_token(tnz,post, **post_param)

    train = Init_trainer(trainer,**train_param)

    tnz.train(files, train)

    if save is True:
        if path is None:
            cpath = cpath.replace(os.sep, '/')
            tnz.save(cpath + "/" + file_name + ".json")
            tokenizer = Tokenizer.from_file(cpath + "/" + file_name + ".json")
        else:
            tnz.save(path +  "/" + file_name + ".json")
            tokenizer = Tokenizer.from_file(path +  "/" + file_name + ".json")
    return tokenizer


# -------------------------------------------------------------------------------------------------------------------------

#[f"C:/Users/thean/Documents/tests/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]
#"C:/Users/thean/Documents/tests/wikitext-103-raw/wiki.test.raw"


token_nlpipe(file_path = ["C:/Users/thean/Documents/tests/wikitext-103-raw/wiki.test.raw"],
             tokenizer = "WordPiece",
             trainer = "WordPieceTrainer",
             pre_tokenizer = "Whitespace",
             Normalizer = [NFD(), Lowercase(), StripAccents()],
             save = True,
             post = "TemplateProcessing",
             path = "C:/Users/thean/Documents/tests/wikitext-103-raw/",
             token_param = {"unk_token": "[UNK]"},
             train_param = {"vocab_size": 30522,
                            "special_tokens": ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]},
             post_param = {"single": "[CLS] $A [SEP]",
                           "pair": "[CLS] $A [SEP] $B:1 [SEP]:1",
                           "special_tokens": [("[CLS]", 1),("[SEP]", 2),]
                           },
)

token_nlpipe(file_path = ["C:/Users/thean/Documents/tests/wikitext-103-raw/wiki.test.raw"],
             tokenizer = "WordPiece",
             default = True,
             save = True,
             path = "C:/Users/thean/Documents/tests/wikitext-103-raw/",
             file_name = "test_tokenizer"
             )

tokenizer1 = Tokenizer.from_file("C:/Users/thean/Documents/tests/wikitext-103-raw/trained_tokenizer.json")
tokenizer2 = Tokenizer.from_file("C:/Users/thean/Documents/tests/wikitext-103-raw/test_tokenizer.json")
output = tokenizer1.encode("Simple is How are you 😁 ?")
print(output.tokens)#
output = tokenizer2.encode("Simple is How are you 😁 ?")
print(output.tokens)