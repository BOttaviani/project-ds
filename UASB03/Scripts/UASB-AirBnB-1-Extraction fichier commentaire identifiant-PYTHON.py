
# coding: utf-8

# # CNAM UASB03 - CERTIFICATION ANALYSE DE DONNEES MASSIVES
# ## Projet d'analyse de sentiment sur les commentaires Airbnb en français
# 
# ***
# Notebook Python de préparation des commentaires en entrée de notre modélisation.
# *  Chargement du fichier brut
# *  Détection de la langue
# *  Filtrage uniquement des commentaires reconnus comme français avec plus de 80% de probabilité
# *  Génération d'un fichier avec uniquement l'identifiant du commentaire (permet de remonter au fichier source) et le commentaire
# 
# 

# In[2]:


import pandas, os, json, numpy
from whatthelang import WhatTheLang
wtl = WhatTheLang()

Airbnb_comment=pandas.read_csv("Data/reviews_Avril2017.csv",encoding='utf-8',lineterminator='\n')
Airbnb_comment.comments.str.replace("\r\n", "")
Airbnb_comment.comments.str.replace("\n", "")
Airbnb_comment.comments.str.encode(encoding='utf-8', errors='strict')
Airbnb_comment['comments']=Airbnb_comment['comments'].astype(str)
Airbnb_comment.head(10)


# In[13]:


Airbnb_comment.shape


# In[5]:


#détection de la langue avec whatthelang
Airbnb_comment['langage']=Airbnb_comment['comments'].apply(wtl.pred_prob)
Airbnb_comment['langue']=Airbnb_comment['langage'].str[0].str[0].str[0]
Airbnb_comment['lg_proba']=Airbnb_comment['langage'].str[0].str[0].str[1]
Airbnb_comment=Airbnb_comment.drop(['listing_id', 'date', 'reviewer_id','reviewer_name','langage'],axis=1)


# In[ ]:


#Récupératoin des commentaire identifiés comme français à plus de 80% de probabilité

Comment = Airbnb_comment[numpy.logical_and(Airbnb_comment['lg_proba']>=0.8,Airbnb_comment['langue']=='fr') ]

Comment['comments'] = Comment['comments'].replace({r'\r\n': ' '}, regex=True).replace("\n", " ").replace({r'\s+$': ' ', r'^\s+': ' '}, regex=True).replace(r'\n',  ' ', regex=True)
Comment = Comment.drop(['langue','lg_proba'],axis=1)


# In[27]:


Comment.shape


# In[28]:


Comment.head()


# In[29]:


# nettoyage des commentaires
Comment['comments']=Comment['comments'].str.strip('\n')
Comment['comments'] = Comment['comments'].astype(str)
Comment['comments'] = Comment['comments'].apply(lambda x: x.replace('.',' ').replace('@',' ').replace('+',' ').
            replace('&',' ').replace(':',' ').replace(':)',' ').replace('\"\"',' ').replace(',',' ').
            replace('(',' ').replace(')',' ').replace(';',' ').replace('!',' ').replace('  ',' '))


# In[30]:


Comment.to_csv("Data/Commentaires_identifiants_V3",encoding='utf-8',header=None,sep='£',index=False)

