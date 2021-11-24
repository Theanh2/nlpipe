#https://huggingface.co/docs/datasets/loading.html
#See docs for load_dataset
#from datasets import load_dataset
#dataset = load_dataset('glue', 'mrpc', split='train')

#-------------------------------------------------------------------------------------------------------------------------

#Normalizer


def norm_list(set, column_n):
    copy_list = set[column_n]
    for i in range(len(copy_list)):
        copy_list[i] = normalizer.normalize_str(copy_list[i])
    print('\n'.join('{}: {}'.format(*k) for k in enumerate(copy_list)))
    #norm_list(dataset, "sentence1")
    return copy_list

#-------------------------------------------------------------------------------------------------------------------------

def decode_token_list(set, column_n):
    copy_list = set[column_n]
    for i in range(len(copy_list)):
        copy_list[i] = tokenizer.decode(copy_list[i])
    #print('\n'.join('{}: {}'.format(*k) for k in enumerate(copy_list)))
    #pretoken_list(dataset, "sentence1")
    return copy_list

#-------------------------------------------------------------------------------------------------------------------------

def enumerate_print(list):
    print('\n'.join('{}: {}'.format(*k) for k in enumerate(list)))

#-------------------------------------------------------------------------------------------------------------------------

from tokenizers import normalizers
from tokenizers.normalizers import NFD,BertNormalizer,Lowercase,NFC,NFD,NFKC,NFKD,Nmt,StripAccents,Strip
def setNormalization(normalise_list):
    """
    Input: List
    BertNormalizer
    Lowercase
    NFC
    NFD
    NFKC
    NFKD
    Nmt
    StripAccents
    Strip
    :return: None, normalizer as global variable
    """
    global normalizer
    normalizer = normalizers.Sequence(normalise_list)


#-------------------------------------------------------------------------------------------------------------------------


