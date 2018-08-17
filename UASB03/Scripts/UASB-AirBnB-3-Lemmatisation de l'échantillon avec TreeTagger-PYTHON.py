
# coding: utf-8

# # CNAM UASB03 - CERTIFICATION ANALYSE DE DONNEES MASSIVES
# ## Projet d'analyse de sentiment sur les commentaires Airbnb en français
# 
# ***
# Notebook Python de lémmatisation du texte sur l'échantillon précedemment constitué.
# 

# In[1]:


import treetaggerwrapper
import pandas
import json


# In[2]:


data = pandas.read_csv("Data/echantillon_evalue.csv",delimiter=',', index_col=0, 
                        encoding='utf-8', usecols=[0, 1, 2])
#data = pandas.read_csv("test_comment_10l.txt",delimiter='#', header=None, index_col=0)
data.columns = ['commentaire', 'qualite']
data.index.names =['Id']
#data['sentiment']=data['sentiment'].astype('category' ,ordered=True)
#data[~data['sentiment'].isin(['positif','neutre', 'négatif'])]


# In[3]:


data.count()


# In[6]:


data['commentaire'] = data['commentaire'].astype(str)
data['qualite'] = data['qualite'].astype(str)

data['commentaire'] = data['commentaire'].apply(lambda x: x.replace('.',' ').replace('@',' ').replace('+',' ').
            replace(',',' ').replace(';',' ').replace('!',' ').replace('  ',' '))


# In[7]:


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


# In[8]:


tagger = treetaggerwrapper.TreeTagger(TAGLANG='fr')

#tag_comment =tagger.tag_text

data['comment_lemm'] = data['commentaire'].apply(comment_to_lemme)
#data.to_csv("comment_tag_Bernard.csv",sep='#')
data['commentaire'] = data['comment_lemm']
data=data.drop(['comment_lemm'],axis=1)
data[data['commentaire'].str.len()>0].to_csv("Data/echantillon_lemmatise.csv")


# ## Signification des codes de la lemmatisation Treetager
# ABR	abreviation
# ADJ	adjective
# ADV	adverb
# DET:ART	article
# DET:POS	possessive pronoun (ma, ta, ...)
# INT	interjection
# KON	conjunction
# NAM	proper name
# NOM	noun
# NUM	numeral
# PRO	pronoun
# PRO:DEM	demonstrative pronoun
# PRO:IND	indefinite pronoun
# PRO:PER	personal pronoun
# PRO:POS	possessive pronoun (mien, tien, ...)
# PRO:REL	relative pronoun
# PRP	preposition
# PRP:det	preposition plus article (au,du,aux,des)
# PUN	punctuation
# PUN:cit	punctuation citation
# SENT	sentence tag
# SYM	symbol
# VER:cond	verb conditional
# VER:futu	verb futur
# VER:impe	verb imperative
# VER:impf	verb imperfect
# VER:infi	verb infinitive
# VER:pper	verb past participle
# VER:ppre	verb present participle
# VER:pres	verb present
# VER:simp	verb simple past
# VER:subi	verb subjunctive imperfect
# VER:subp	verb subjunctive present
