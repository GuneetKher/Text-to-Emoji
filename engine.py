import numpy as np
import emoji
import spacy
import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import norm
from IPython.core.display import display, HTML
import sys
from flask import Flask

app = Flask(__name__)




from emo_uni import emo_list,emo_get

e_l=[]
for i in emo_list:
    e_l.append(str(i.replace("_"," ")).lower())

nlp = spacy.load('en_core_web_sm')
from tqdm import tqdm
with open('glove.6B.300d.txt', 'r',encoding="utf8") as f:
    for line in tqdm(f, total=400000):
        parts = line.split()
        word = parts[0]
        vec = np.array([float(v) for v in parts[1:]], dtype='f')
        nlp.vocab.set_vector(word, vec)

docs = [nlp(str(keywords)) for keywords in tqdm(e_l)]
doc_vectors = np.array([doc.vector for doc in docs])

def most_similar(vectors, vec):
    cosine = lambda v1, v2: dot(v1, v2) / (norm(v1) * norm(v2))
    dst = np.dot(vectors, vec) / (norm(vectors) * norm(vec))
    return (np.argsort(-dst))[0],max(dst)


sentences=["star boy","i love you","x marks the spot","the pizza is great","chicken lays eggs","games or a mobile ?","i have scored hundred in maths","She is the queen of hearts","messi is the king of soccer","lets build a rocket","I will dance at your wedding","i love my mother","coffee or tea","aliens have cool spaceships","india is the greatest","china is a wonderful country","good job","the terrorist bombed the church"]

@app.route("/convert/<sentence>")
def convert(sentence):
    for sentence in sentences:
        l=[]
        for w in sentence.split(" "):
            v = nlp(w.lower()).vector
            ms,sim=most_similar(doc_vectors, v)
            #print(sim)
            if(sim>0.0115):
                word=emo_get[ms]
                l.append(emoji.emojize(word,use_aliases=True))
            else:
                l.append(w)
        return ''.join([x for x in l])
        # display(HTML('<font size="+3">{}</font>'.format(' '.join([x for x in l]))))

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000)


# sentence='star boy'
# l=[]
# for w in sentence.split(" "):
#     v = nlp(w.lower()).vector
#     ms,sim=most_similar(doc_vectors, v)
#         #print(sim)
#     if(sim>0.0115):
#         word=emo_get[ms]
#         l.append(emoji.emojize(word,use_aliases=True))
#     else:
#         l.append(w)
#     print(l)