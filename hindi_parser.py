import pickle
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize

remove_nuktas = False
factory = IndicNormalizerFactory()
normalizer = factory.get_normalizer("hi", remove_nuktas)

hin = open('./model/dataset/en-hi.hi').readlines()
hin = [line.decode('UTF-8') for line in hin]
print(hin[:5])
hin = [normalizer.normalize(line.strip()) for line in hin]

hin = [indic_tokenize.trivial_tokenize(line) for line in hin]
print(hin[:5])

with open("hindi_tokens.txt", "wb") as fp:
    pickle.dump(hin, fp)
