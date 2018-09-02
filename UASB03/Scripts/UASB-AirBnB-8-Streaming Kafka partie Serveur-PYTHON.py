
# coding: utf-8

# # CNAM UASB03 - CERTIFICATION ANALYSE DE DONNEES MASSIVES
# ## Projet d'analyse de sentiment sur les commentaires Airbnb en français
# 
# ***
# Notebook Python simulation d'envoi de commentaires du site vers un script client charger de faire l'analyse.
# Le traitement effectue les étapes suivantes :
# * détection de la langue
# * lemmatisation pour les commentaires identifiés comme français
# * envoi des données à 3 Topics Kafka selon la langue
# 

# -  ## Import des librairies et chargement du fichier

# In[1]:


import time
from random import randint
import json, pandas
from kafka import SimpleProducer, KafkaClient
from kafka import KafkaProducer
import treetaggerwrapper
tagger = treetaggerwrapper.TreeTagger(TAGLANG='fr')

from whatthelang import WhatTheLang
wtl = WhatTheLang()

producer = KafkaProducer(bootstrap_servers='localhost:9092',
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'), linger_ms=10)

# Temps d'attente maximale fixé à 10 secondes
ATTENTE_MAX = 1

fichier = open('Data/new_comment_2018.json', 'rb')
#fichier = open('Data/comment_2018_echantillon.json', 'rb')
lignes = fichier.readlines()

def comment_to_lemme(comment):
    t=treetaggerwrapper.make_tags(tagger.tag_text(comment))
    lemme=''
    for i in t:
        if type(i)==treetaggerwrapper.Tag:
            if i.pos[:3] in ('ADJ', 'ADV', 'INT','KON','NOM','VER'): 
                if i.lemma !='dns-remplacé':
                    if len(i.lemma)>1 :
                        lemme =lemme+' '+i.lemma.split('|')[0].lower()
    return lemme


# -  ## Transfert des données

# In[ ]:


while True:
   
    try:
        print('\nEnvoi des données...\n')
        i = 0
        nb_msg = 0
        while(i < len(lignes)):
            comment = pandas.read_json(lignes[i].decode("utf8").encode('utf8'),typ='series').to_frame().transpose()
            comment['comments']=comment['comments'].replace("\r\n", "")
            comment['comments']=comment['comments'].replace("\n", "")
            comment['comments']=comment['comments'].astype(str)
            # Détection de la langue et ajout au dataframe
            comment['langage']=comment['comments'].apply(wtl.pred_prob)
            comment['langage'] =comment['langage'].astype(str).apply(lambda x: x.replace('[','').replace(']','').
                                replace('(','[').replace(')',']')).apply(lambda x: x[1:-1].split(','))
            comment['langue']=comment['langage'].str[0]
            langue = comment['langue'].iloc[0].replace('\'','')
            print(langue)
            comment['lg_proba']=comment['langage'].str[1].astype(float)
            proba = comment['lg_proba'].iloc[0]
            comment=comment.drop(['langage'],axis=1)
            comment['comments'] = comment['comments'].apply(lambda x: x.replace('.',' ').replace('@',' ').
                           replace('+',' ').replace(',',' ').replace(';',' ').replace('!',' ').replace('  ',' '))
            if (proba>0.8) :
                if langue=='fr':
                    comment['comment_length']=comment['comments'].astype(str).str.len()
                    comment['comment_lemm'] = comment['comments'].apply(comment_to_lemme)
                    producer.send('AirBnb_income_fr',value=comment.to_dict('records'))
                elif  langue=='en':
                    producer.send('AirBnb_income_en',value=comment.to_dict('records'))
                else:
                    producer.send('AirBnb_income_other',value=comment.to_dict('records'))
            producer.flush()
            
            i = i + 1
            nb_msg = nb_msg + 1
            attente = randint(0, ATTENTE_MAX)
            print ("Attente " + str(attente) + " secondes ")
            print("Nombre de messages envoyés : "+str(nb_msg))
            time.sleep( attente)
        break
    except Exception as ex:
        print (ex)

