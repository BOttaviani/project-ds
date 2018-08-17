
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.RegexTokenizer

//LECTURE DU FICHIER D'ENTREE DE L'ECHANTILLON DE COMMENTAIRES EVALUES
val echantillon = spark.read.format("csv").option("header", "true").load("Data/echantillon_lemmatise.csv")
echantillon.createOrReplaceTempView("echantillon")
val echantillonDF = spark.sql("select cast(Id_comment as Int) as Id, concat(qualite, '$ ',commentaire) as commentaire_enrichi from echantillon")

//DECOUPAGE DU CORPUS DE COMMENTAIRES EN MOTS
//Définition des paramètres du découpage des commentaires en mots
val tokenizer = new RegexTokenizer()
  .setPattern(" ") // Séparateur entre les mots du commentaire
  .setMinTokenLength(4) // Filtre tous les mots du commentaire de longueur <= 4
  .setInputCol("commentaire_enrichi")
  .setOutputCol("mots")
//Découpage de chaque commentaire en mots
val tokenized_df = tokenizer.transform(echantillonDF)

//FILTRAGE DES STOPS WORDS
//Lecture du fichier de stop words
val stopwords = sc.textFile("Data/French_stop_words").collect()
//Définition des paramètres du filtre à stop words
val remover = new StopWordsRemover()
  .setStopWords(stopwords) // This parameter is optional
  .setInputCol("mots")
  .setOutputCol("mots_filtres")
//Création d'un dataframe des mots filtrés
val filtrage_df = remover.transform(tokenized_df)
//Création d'un dataframe avec les mots filtres uniqument
val echantillon_filtreWArray = filtrage_df.select("mots_filtres")

//Remise au format String de la zone "mots_filtres"
val echantillon_filtreDF = echantillon_filtreWArray.as[Array[String]]
  .map { case (echantillon_filtreWArray) => (echantillon_filtreWArray.mkString(" ")) }
  .toDF("Liste_mots")

echantillon_filtreDF.coalesce(1).write.format("com.databricks.spark.csv").option("header","true").save("Data/echantillon_lemmatise_SWR")

//LECTURE DU FICHIER D'ENTREE DE L'ECHANTILLON DE COMMENTAIRES EVALUES
val echantillon = spark.read.format("csv").option("header", "true").load("Data/echantillon_evalue.csv")
echantillon.createOrReplaceTempView("echantillon")
val echantillonDF = spark.sql("select cast(Id_comment as Int) as Id, concat(qualite, '$ ',commentaire) as commentaire_enrichi from echantillon")

//DECOUPAGE DU CORPUS DE COMMENTAIRES EN MOTS
//Définition des paramètres du découpage des commentaires en mots
val tokenizer = new RegexTokenizer()
  .setPattern(" ") // Séparateur entre les mots du commentaire
  .setMinTokenLength(4) // Filtre tous les mots du commentaire de longueur <= 4
  .setInputCol("commentaire_enrichi")
  .setOutputCol("mots")
//Découpage de chaque commentaire en mots
val tokenized_df = tokenizer.transform(echantillonDF)

//FILTRAGE DES STOPS WORDS
//Lecture du fichier de stop words
val stopwords = sc.textFile("Data/French_stop_words").collect()
//Définition des paramètres du filtre à stop words
val remover = new StopWordsRemover()
  .setStopWords(stopwords) // This parameter is optional
  .setInputCol("mots")
  .setOutputCol("mots_filtres")
//Création d'un dataframe des mots filtrés
val filtrage_df = remover.transform(tokenized_df)
//Création d'un dataframe avec les mots filtres uniqument
val echantillon_filtreWArray = filtrage_df.select("mots_filtres")

//Remise au format String de la zone "mots_filtres"
val echantillon_filtreDF = echantillon_filtreWArray.as[Array[String]]
  .map { case (echantillon_filtreWArray) => (echantillon_filtreWArray.mkString(" ")) }
  .toDF("Liste_mots")

echantillon_filtreDF.coalesce(1).write.format("com.databricks.spark.csv").option("header","true").save("Data/echantillon_brut_SWR")

//VECTORISATION DES DONNEES AVEC HASHING TF
import org.apache.spark.sql.SparkSession
import org.apache.spark._
import org.apache.spark.rdd._
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.regression.LabeledPoint
import scala.util.{Success, Try}
import org.apache.spark.mllib.util.MLUtils

//LECTURE DU FICHIER D'ENTREE DE L'ECHANTILLON DE COMMENTAIRES EVALUES
val echantillon_filtreDF = spark.read.format("csv").option("header", "true").load("Data/echantillon_lemmatise_SWR")

//CONSTRUCTION DU FORMAT LIBSVM
//Variables d'entrée
val echantillonRDD = echantillon_filtreDF.rdd
val hashingTF = new HashingTF(12000)
//Constitution du message avec le label
val echantillon_reformat = echantillonRDD.map(
  row =>{
    Try{
      val msg = row.toString.toLowerCase()
      var isHappy:Int = 0
      if(msg.contains("positif$")){
        isHappy = 1
      }else if(msg.contains("negatif$")){
        isHappy = 0
      }
      var msgSanitized = msg.replace("positif$ ", "")
      msgSanitized = msgSanitized.replace("negatif$ ","")
      //Return a tuple
      (isHappy, msgSanitized.split(" ").toSeq)
    }
  }
)
var echantillon_ok = echantillon_reformat.filter((_.isSuccess)).map(_.get)
//Transformation du texte en nombre par application de la fonction de hachage et mise au format LIBSVM
val echantillon_transfo = echantillon_ok.map(
  t => (t._1, hashingTF.transform(t._2)))
  .map(x => new LabeledPoint((x._1).toDouble, x._2))

//ECRITURE EN FICHIER DE SORTIE AU FORMAT LIBSVM
MLUtils.saveAsLibSVMFile(echantillon_transfo,"Data/Vecto_HTF")

// VECTORISATION AVEC WORD2VEC
import org.apache.spark.sql.SparkSession
import org.apache.spark._
import org.apache.spark.rdd._
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.regression.LabeledPoint
import scala.util.{Success, Try}
import org.apache.spark.mllib.linalg.{Vector, Vectors, DenseVector, SparseVector}
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}
import org.apache.spark.mllib.linalg.{Vector => SparkVector}
import org.apache.spark.mllib.util.MLUtils
def toBreeze(v:SparkVector) = BV(v.toArray)
def fromBreeze(bv:BV[Double]) = Vectors.dense(bv.toArray)
def add(v1:SparkVector, v2:SparkVector) = fromBreeze(toBreeze(v1) + toBreeze(v2))
def scalarMultiply(a:Double, v:SparkVector) = fromBreeze(a * toBreeze(v))

//LECTURE DU FICHIER D'ENTREE DE L'ECHANTILLON DE COMMENTAIRES EVALUES
val echantillon_filtreDF = spark.read.format("csv").option("header", "true").load("Data/echantillon_brut_SWR")

//CHARGEMENT DU MODELE WORD2VEC
//val word2vec = new Word2Vec()
val w2vModel = Word2VecModel.load(sc, "modele/Word2VecFR_complet")

//MISE EN FORME DU MODELE LU
val vectors = w2vModel.getVectors.mapValues(vv => Vectors.dense(vv.map(_.toDouble))).map(identity)

// transmettre la map aux noeuds de calcul
val bVectors = sc.broadcast(vectors)

//TAILLE DES VECTEURS WORD2VEC
val vectSize = 100

//CONSTRUCTION DES VECTEURS ASSOCIES AUX COMMENTAIRES DE L'ECHANTILLON
val comment2vec_comptage = echantillon_filtreDF.rdd.map{ row => (row.getAs[String](0)) }.filter(sentence => sentence.length >= 1)
    .map(sentence => sentence.toLowerCase.split(" "))
    .map(wordSeq => { 
                     Try {                    
                          var isHappy:Int = -1
                          var vSum = Vectors.zeros(vectSize)
                          var vNb = 0
                          wordSeq.foreach {word =>
                                           if(word.length >= 2) {
                                                                 bVectors.value.get(word).foreach {v =>
                                                                                                   vSum = add(v, vSum)
                                                                                                   vNb += 1
                                                                                                  }
                                                                }
                                           if(word.contains("positif$")){isHappy = 1}
                                           else if(word.contains("negatif$")){isHappy = 0}
                                          }
                          if (vNb != 0) {
                                         vSum = scalarMultiply(1.0 / vNb, vSum)
                                        }
                          (isHappy, vSum, vSum.numNonzeros)
                         }
                       }
                     )

//FILTRAGE DES VECTEURS NULS
//Création du dataframe associé aux vecteurs Word2vec
val echantillon_comptage = comment2vec_comptage.filter(_.isSuccess).map(_.get).toDF("Label", "Vecteur", "Nb")
//Création de la table associée aux vecteurs Word2vec
echantillon_comptage.createOrReplaceTempView("comptage_vecteur")
//Récupération des vecteurs non nuls
val echantillon_filtre = spark.sql("select Label, Vecteur from comptage_vecteur where Nb > 0")
//Mise au format RDD
val echantillon_filtreRDD = echantillon_filtre.rdd.map(x => (x.getAs[Int](0), x.getAs[Vector](1)))

//CONVERSION AU FORMAT LIBSVM
val echantillon_transfo = echantillon_filtreRDD.map(x => new LabeledPoint((x._1).toDouble, x._2))

//ECRITURE EN FICHIER DE SORTIE AU FORMAT LIBSVM
MLUtils.saveAsLibSVMFile(echantillon_transfo,"Data/Vecto_Word2vec")

// VECTORISATION AVEC WORD2VEC
import org.apache.spark.sql.SparkSession
import org.apache.spark._
import org.apache.spark.rdd._
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.regression.LabeledPoint
import scala.util.{Success, Try}
import org.apache.spark.mllib.linalg.{Vector, Vectors, DenseVector, SparseVector}
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}
import org.apache.spark.mllib.linalg.{Vector => SparkVector}
import org.apache.spark.mllib.util.MLUtils
def toBreeze(v:SparkVector) = BV(v.toArray)
def fromBreeze(bv:BV[Double]) = Vectors.dense(bv.toArray)
def add(v1:SparkVector, v2:SparkVector) = fromBreeze(toBreeze(v1) + toBreeze(v2))
def scalarMultiply(a:Double, v:SparkVector) = fromBreeze(a * toBreeze(v))

//LECTURE DU FICHIER D'ENTREE DE L'ECHANTILLON DE COMMENTAIRES EVALUES
val echantillon_filtreDF = spark.read.format("csv").option("header", "true").load("Data/echantillon_brut_SWR")

//CHARGEMENT DU MODELE WORD2VEC
//val word2vec = new Word2Vec()
val w2vModel = Word2VecModel.load(sc, "modele/Word2VecFR_comment")

//MISE EN FORME DU MODELE LU
val vectors = w2vModel.getVectors.mapValues(vv => Vectors.dense(vv.map(_.toDouble))).map(identity)

// transmettre la map aux noeuds de calcul
val bVectors = sc.broadcast(vectors)

//TAILLE DES VECTEURS WORD2VEC
val vectSize = 100

//CONSTRUCTION DES VECTEURS ASSOCIES AUX COMMENTAIRES DE L'ECHANTILLON
val comment2vec_comptage = echantillon_filtreDF.rdd.map{ row => (row.getAs[String](0)) }.filter(sentence => sentence.length >= 1)
    .map(sentence => sentence.toLowerCase.split(" "))
    .map(wordSeq => { 
                     Try {                    
                          var isHappy:Int = -1
                          var vSum = Vectors.zeros(vectSize)
                          var vNb = 0
                          wordSeq.foreach {word =>
                                           if(word.length >= 2) {
                                                                 bVectors.value.get(word).foreach {v =>
                                                                                                   vSum = add(v, vSum)
                                                                                                   vNb += 1
                                                                                                  }
                                                                }
                                           if(word.contains("positif$")){isHappy = 1}
                                           else if(word.contains("negatif$")){isHappy = 0}
                                          }
                          if (vNb != 0) {
                                         vSum = scalarMultiply(1.0 / vNb, vSum)
                                        }
                          (isHappy, vSum, vSum.numNonzeros)
                         }
                       }
                     )

//FILTRAGE DES VECTEURS NULS
//Création du dataframe associé aux vecteurs Word2vec
val echantillon_comptage = comment2vec_comptage.filter(_.isSuccess).map(_.get).toDF("Label", "Vecteur", "Nb")
//Création de la table associée aux vecteurs Word2vec
echantillon_comptage.createOrReplaceTempView("comptage_vecteur")
//Récupération des vecteurs non nuls
val echantillon_filtre = spark.sql("select Label, Vecteur from comptage_vecteur where Nb > 0")
//Mise au format RDD
val echantillon_filtreRDD = echantillon_filtre.rdd.map(x => (x.getAs[Int](0), x.getAs[Vector](1)))

//CONVERSION AU FORMAT LIBSVM
val echantillon_transfo = echantillon_filtreRDD.map(x => new LabeledPoint((x._1).toDouble, x._2))

//ECRITURE EN FICHIER DE SORTIE AU FORMAT LIBSVM
MLUtils.saveAsLibSVMFile(echantillon_transfo,"Data/Vecto_Word2vecC2")

import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.sql.functions._
import scala.util.{Success, Try}
import org.apache.spark.mllib.linalg.{Vector, Vectors, DenseVector, SparseVector}
import org.apache.spark.mllib.linalg.{Vector => SparkVector}
import org.apache.spark.mllib.util.MLUtils

//LECTURE DU FICHIER D'ENTREE DE L'ECHANTILLON DE COMMENTAIRES EVALUES FILTRE AU FORMAT STRING
val echantillon_filtreDF = spark.read.format("csv").option("header", "true").load("Data/echantillon_lemmatise_SWR")

//LECTURE DU FICHIER D'ENTREE DE L'ECHANTILLON DE COMMENTAIRES EVALUES FILTRE AU FORMAT ARRAY[STRING]
val sqlContext = new org.apache.spark.sql.SQLContext(sc)
val echantillon_filtreDF_Array = sqlContext
  .read
  .format("com.databricks.spark.csv")
  .option("header", true)
  .csv("Data/echantillon_filtre_SWR_V2")
  .withColumn("Liste_mots", split($"Liste_mots", " "))


val echantillonRDD = echantillon_filtreDF.rdd
//Constitution du message avec le label
val echantillon_reformat = echantillonRDD.map(
  row =>{
    Try{
      val msg = row.toString.toLowerCase()
      var isHappy:Int = 0
      if(msg.contains("positif$")){
        isHappy = 1
      }else if(msg.contains("negatif$")){
        isHappy = 0
      }
      var msgSanitized = msg.replace("positif$ ", "")
      msgSanitized = msgSanitized.replace("negatif$ ","")
      //Return a tuple
      (isHappy, msgSanitized.split(" ").toSeq)
    }
  }
)
//Filtrage pour ne garder que les utils de type "is_success" et conversion au format RDD
var echantillon_ok = echantillon_reformat.filter(_.isSuccess).map(_.get)
//Mise au format dataframe avec les bons noms de colonne
val echantillon_ok_DF = echantillon_ok.toDF("Label", "Liste_mots")
//VECTORISATION PAR COUNTVECTORIZER
//Définition des paramètres pour l'outil de vectorisation
//setMinDF: Paramètre Specifiant le nombre minimum de documents different dans lesquels un mot doit apparaître pour être pris en compte.
val vectorizer = new CountVectorizer()
  .setInputCol("Liste_mots")
  .setOutputCol("vecteurs")
  .setVocabSize(10000)
  .setMinDF(5)
  .fit(echantillon_ok_DF)

//Sauvegarde du modèle pour réutilisation ultérieure
vectorizer.save("modele/CountVectorizer")
//Transformation des mots en vecteurs
val echantillon_vect = vectorizer.transform(echantillon_ok_DF)
echantillon_vect.createOrReplaceTempView("echantillon_vect")
//Conversion du format Integer au format Double pour le label
val echantillon_vect_conv = spark.sql("select cast(Label as Double) as label, vecteurs from echantillon_vect")
//Mise au format LIBSVM
val echantillon_transfo = echantillon_vect_conv.select("label","vecteurs").rdd.map(row => LabeledPoint(
                                                             row.getAs[Double]("label"),
                                                             org.apache.spark.mllib.linalg.Vectors.fromML(row.getAs[org.apache.spark.ml.linalg.SparseVector]("vecteurs"))
                                                                                                      ))

//ECRITURE EN FICHIER DE SORTIE AU FORMAT LIBSVM
MLUtils.saveAsLibSVMFile(echantillon_transfo,"Data/Vecto_Countvectorizer")

