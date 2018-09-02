
import kafka.serializer.StringDecoder
import org.apache.spark.sql.SparkSession
import org.apache.spark._
import org.apache.spark.rdd._
import org.apache.spark.SparkContext._
import org.apache.spark.streaming._
import org.apache.spark.streaming.kafka._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions.concat_ws
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.regression.LabeledPoint
import scala.util.{Success, Try}
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.mllib.linalg.{Vector, Vectors, DenseVector, SparseVector}
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}
import org.apache.spark.mllib.linalg.{Vector => SparkVector}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
def toBreeze(v:SparkVector) = BV(v.toArray)
def fromBreeze(bv:BV[Double]) = Vectors.dense(bv.toArray)
def add(v1:SparkVector, v2:SparkVector) = fromBreeze(toBreeze(v1) + toBreeze(v2))
def scalarMultiply(a:Double, v:SparkVector) = fromBreeze(a * toBreeze(v))

import com.mongodb.spark.sql._
import org.bson.Document

import org.apache.kafka.clients.producer._
import java.util.Properties

import java.util


// Paramétrage de la session Spark
val spark = SparkSession.builder()
      .master("local[3]")
      .appName("MongoSparkConnectorIntro")
      .config("spark.mongodb.input.uri", "mongodb://127.0.0.1:27017/airbnb.Commentaires_evalues")
      .config("spark.mongodb.output.uri", "mongodb://127.0.0.1:27017/airbnb.Commentaires_evalues")
      .getOrCreate()
sc.setLogLevel("ERROR")
val sqlContext = new SQLContext(sc)
@transient val ssc = new StreamingContext(sc, Minutes(1))

// Définition du Consumer Kakka ayant les messages en entrée du Streaming
val topicsSet = "AirBnb_income_fr".split(",").toSet
val kafkaParams = Map[String, String]("metadata.broker.list" -> "localhost:9092")
@transient val messages = KafkaUtils.createDirectStream[String, String, StringDecoder, StringDecoder](
      ssc, kafkaParams, topicsSet
    )
// Définition des propriété du Producer Kafka pour enregistré les messages insatisfaits
val props = new Properties()
    props.put("bootstrap.servers", "localhost:9092")
    props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer")
    props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer")

// CHARGEMENT DES STOPWORDS
val stopwords = sc.textFile("Data/French_stop_words").collect()

//CHARGEMENT DES MODELES SPARK MLIB PRE-CALCULE EN PHASE DE MODELISATION
val SVMModel_HTF = SVMModel.load(sc, "modele/SVM_HTF/")
val SVMModel_W2V = SVMModel.load(sc, "modele/SVM_W2VC2/")


messages.foreachRDD { rdd =>  

    //définition du format Json en entrée du streaming
    val schema = new StructType()
      .add("comment_lemm", StringType)
      .add("comment_length", LongType)
      .add("comments", StringType)
      .add("date", LongType)
      .add("id", LongType)
      .add("langue", StringType)
      .add("lg_proba", DoubleType)
      .add("listing_id", LongType)
      .add("reviewer_id", LongType)
      .add("reviewer_name", StringType)

 //Lecture du rdd avec le schéma Json dans un dataframe
val df = sqlContext.read.schema(schema).json(rdd.map(x => x._2).toDS)

df.createOrReplaceTempView("df")

//séparation des messages de +/- de 300 caractères    
val df_under_300 = spark.sql("select * from df where comment_length <=300")
val df_over_300 = spark.sql("select * from df where comment_length >300")

//Tokenization du message initial pour les moins de 300 caractères    
val tokenizer_under_300 = new RegexTokenizer()
  .setPattern(" ") // Séparateur entre les mots du commentaire
  .setMinTokenLength(4) // Filtre tous les mots du commentaire de longueur <= 4
  .setInputCol("comments")
  .setOutputCol("mots")
val tokenized_under_300 = tokenizer_under_300.transform(df_under_300)

//Tokenization du message lemmatisé pour les plus de 300 caractères    
val tokenizer_over_300 = new RegexTokenizer()
  .setPattern(" ") // Séparateur entre les mots du commentaire
  .setMinTokenLength(4) // Filtre tous les mots du commentaire de longueur <= 4
  .setInputCol("comment_lemm")
  .setOutputCol("mots")
val tokenized_over_300 = tokenizer_over_300.transform(df_over_300)

// Suppression des StopWords dans les 2 cas
val remover = new StopWordsRemover()
  .setStopWords(stopwords) 
  .setInputCol("mots")
  .setOutputCol("mots_filtres")

//Création des dataframe des mots filtrés
val filtrage_under_300 = remover.transform(tokenized_under_300)
val filtrage_over_300 = remover.transform(tokenized_over_300)

//Vectorisation HashingTF pour les plus de 300 caractères
val hashingTF = new HashingTF()
   .setNumFeatures(12000)
   .setInputCol("mots_filtres")
   .setOutputCol("features")
    
val hashingTF_over_300 = hashingTF.transform(filtrage_over_300)
val hashingTF_transfo = MLUtils.convertVectorColumnsFromML(hashingTF_over_300, "features")  
val feature_HTF = hashingTF_transfo.select("features").rdd

// Application du modèle SVM pour HashingTF sur les plus de 300 caratères
val prediction_SVM_HTF = SVMModel_HTF.predict(feature_HTF.map(row =>row.getAs[org.apache.spark.mllib.linalg.Vector]("features")))

// Vectorisation Word2Vec pour les moins de 300 caractères
//CHARGEMENT DU MODELE WORD2VEC
val w2vModel = Word2VecModel.load(sc,"modele/Word2VecFR_comment")
// Transmission des vecteurs Word2Vec aux noeuds de calcul
//MISE EN FORME DU MODELE LU
val vectors = w2vModel.getVectors.mapValues(vv => Vectors.dense(vv.map(_.toDouble))).map(identity)
val bVectors = sc.broadcast(vectors)
//TAILLE DES VECTEURS WORD2VEC
val vectSize = 100
// TRANSFORMATION DE LA LISTE DES MOTS FILTRES EN STRING
filtrage_under_300.select("mots_filtres").createOrReplaceTempView("test")
val text_under_300=sqlContext.sql("select concat_ws(\" \",mots_filtres) as mots_filtres from test").rdd
//CONSTRUCTION DES VECTEURS ASSOCIES AUX COMMENTAIRES DE L'ECHANTILLON
val comment2vec_comptage = text_under_300.map{ row => (row.getAs[String](0)) }.filter(sentence => sentence.length >= 1)
    .map(sentence => sentence.toLowerCase.split(" "))
    .map(wordSeq => { 
                     Try {                    
                          var vSum = Vectors.zeros(vectSize)
                          var vNb = 0
                          wordSeq.foreach {word =>
                                           if(word.length >= 2) {
                                                                 bVectors.value.get(word).foreach {v =>
                                                                                                   vSum = add(v, vSum)
                                                                                                   vNb += 1
                                                                                                  }
                                                                }
                                          }
                          if (vNb != 0) {
                                         vSum = scalarMultiply(1.0 / vNb, vSum)
                                        }
                          ( vSum, vSum.numNonzeros)
                         }
                       }
                     )
//FILTRAGE DES VECTEURS NULS
val echantillon_comptage = comment2vec_comptage.filter(_.isSuccess).map(_.get).toDF( "Vecteur", "Nb")
echantillon_comptage.createOrReplaceTempView("comptage_vecteur")
//Récupération des vecteurs non nuls
val echantillon_filtre = spark.sql("select Vecteur from comptage_vecteur where Nb > 0")

val V_W2V = echantillon_filtre.select("Vecteur").rdd

// Application du modèle SVM pour Word2Vecc Corpus2 sur les moins de 300 caratères 
val prediction_SVM_W2V = SVMModel_W2V.predict(V_W2V.map(row =>row.getAs[org.apache.spark.mllib.linalg.Vector]("Vecteur")))

// jointure de la prédiction avec le dataframe d'origine pour les - de 300
val df_under_300_2 = sqlContext.createDataFrame(
  df_under_300.rdd.zipWithIndex.map{case (row, columnindex) => Row.fromSeq(row.toSeq :+ columnindex)},
  StructType(df_under_300.schema.fields :+ StructField("columnindex", LongType, false))
)
val df_pred_W2V=sqlContext.createDataFrame(
  prediction_SVM_W2V.toDF().rdd.zipWithIndex.map{case (row, columnindex) => Row.fromSeq(row.toSeq :+ columnindex)},
  StructType(prediction_SVM_W2V.toDF().schema.fields :+ StructField("columnindex", LongType, false))
)
val result_under_300 = df_under_300_2.join(df_pred_W2V, Seq("columnindex")).drop("columnindex")

// jointure de la prédiction avec le dataframe d'origine pour les + de 300
val df_over_300_2 = sqlContext.createDataFrame(
  df_over_300.rdd.zipWithIndex.map{case (row, columnindex) => Row.fromSeq(row.toSeq :+ columnindex)},
  StructType(df_over_300.schema.fields :+ StructField("columnindex", LongType, false))
)
val df_pred_HTF=sqlContext.createDataFrame(
  prediction_SVM_HTF.toDF().rdd.zipWithIndex.map{case (row, columnindex) => Row.fromSeq(row.toSeq :+ columnindex)},
  StructType(prediction_SVM_HTF.toDF().schema.fields :+ StructField("columnindex", LongType, false))
)
val result_over_300 = df_over_300_2.join(df_pred_HTF, Seq("columnindex")).drop("columnindex")

// Union des résultats des 2 modèles appliqués
val result=result_over_300.union(result_under_300)
result.show()
//Sauvegarde du résultat dans une base MongoDB
result.saveToMongoDB()

result.createOrReplaceTempView("result")

// Sélection des insatisfaits
val insatisfaits = spark.sql("select * from result where value=0")

//Envoi des insatisfaits dans le Topik Kafka AirBnb_insatisfaits_fr
insatisfaits.selectExpr("to_json(struct(*)) AS value").foreach{row=>{
  val producer = new KafkaProducer[String, String](props)
  val kMessage=new ProducerRecord[String,String]("AirBnb_insatisfaits_fr",row.getString(0))
 producer.send(kMessage)
 }}

    
};
ssc.start()

ssc.stop()

sc.stop()
