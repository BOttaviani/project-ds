
// Import des librairies
import org.apache.spark.sql.SparkSession
import org.apache.spark._
import org.apache.spark.rdd._
import org.apache.spark.SparkContext._
import org.apache.spark.ml._
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

import scala.util.{Success, Try}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.StopWordsRemover

//LECTURE DU FICHIER D'ENTREE DES COMMENTAIRES AVEC DELIMITEUR #
val commentaires_bruts = spark.read.option("delimiter", "#")
                                    .option("multiline",true)
                                    .csv("Data/commentaires_evalues_V2.txt")

val stopwords = spark.sparkContext.textFile("French_stop_words.txt").collect()
val remover = new StopWordsRemover()
      .setStopWords(stopwords)
      .setInputCol("_c1")
      .setOutputCol("removed")

val toArray = udf[Array[String], String]( _.split("[ '-.,;:?!\\(\\)]"))
val commentaires_toarray = commentaires_bruts.withColumn("_c1", toArray(lower(commentaires_bruts("_c1"))))
val commentaires_withoutSW = remover.transform(commentaires_toarray)  

//CREATION DE LA TABLE ASSOCIEE AUX COMMENTAIRES LUS
commentaires_withoutSW.createOrReplaceTempView("commentaires_withoutSW")

//CREATION DU NOM DES COLONNES POUR LA TABLE COMMENTAIRES
val commentaires = spark.sql("select _c0 as Id_comment, concat_ws(' ',_c1) as commentaire,concat_ws(' ',removed) as commentaire_SW, _c2 as qualite from commentaires_withoutSW")

//CREATION DE LA TABLE ASSOCIEE AUX COMMENTAIRES FINALISES
commentaires.createOrReplaceTempView("commentaires")


commentaires.take(5)

val commentaires_negatifs = spark.sql("select * from commentaires where qualite like '%négatif%'") 
val commentaires_positifs = spark.sql("select * from commentaires where qualite like '%positif%'")
val commentaires_positifs_nbre = commentaires_positifs.count()
val commentaires_negatifs_nbre = commentaires_negatifs.count()


//DEFINITION DE LA TAILLE DE L'ECHANTILLON D'APPRENTISSAGE
val taille_echantillon = Math.min(commentaires_negatifs_nbre, commentaires_positifs_nbre).toInt
//DEFINITION DE L'ECHANTILLON D'APPRENTISSAGE AVEC AUTANT DE COMMENTAIRES POSITIFS QUE NEGATIFS
var echantillon = commentaires_positifs.limit(taille_echantillon).unionAll(commentaires_negatifs.limit(taille_echantillon))
echantillon.createOrReplaceTempView("echantillon")
val echantillon_msg = spark.sql("select concat( qualite, ' ', commentaire_SW) as msg from echantillon")
val echantillonRDD = echantillon_msg.rdd
val hashingTF = new HashingTF(20000)


println("Commentaires                  = "+commentaires.count())
println("   dont commentaires négatifs = "+commentaires_negatifs.count())
println("   dont commentaires positifs = "+commentaires_positifs.count())
println("Taille echantillon = "+echantillonRDD.count())


val echantillon_reformat = echantillonRDD.map(
  row =>{
    Try{
      val msg = row.toString.toLowerCase()
      var isHappy:Int = 0
      if(msg.contains("positif ")){
        isHappy = 1
      }else if(msg.contains("negatif ")){
        isHappy = 0
      }
      var msgSanitized = msg.replaceAll("positif ", "")
      msgSanitized = msgSanitized.replaceAll("negatif ","")
      //Return a tuple
      (isHappy, msgSanitized.split(" ").toSeq)
    }
  }
)

var echantillon_ok = echantillon_reformat.filter((_.isSuccess)).map(_.get)
val echantillon_transfo = echantillon_ok.map(
  t => (t._1, hashingTF.transform(t._2)))
  .map(x => new LabeledPoint((x._1).toDouble, x._2))

var sample = echantillon_ok.take(1000).map(
  t => (t._1, hashingTF.transform(t._2), t._2))
  .map(x => (new LabeledPoint((x._1).toDouble, x._2), x._3))

val splits = echantillon_transfo.randomSplit(Array(0.7, 0.3))
val (trainingData, validationData) = (splits(0), splits(1))


//CONSTRUCTION DU MODELE

val boostingStrategy = BoostingStrategy.defaultParams("Classification")
boostingStrategy.setNumIterations(20) //number of passes over our training data
boostingStrategy.treeStrategy.setNumClasses(2) //We have two output classes: happy and sad
boostingStrategy.treeStrategy.setMaxDepth(5)

val model = GradientBoostedTrees.train(trainingData, boostingStrategy)

// Sauvegarde du modèle
model.save(sc, "modele/GradientBoosting")

val predictionAndLabel = validationData.map(p => (model.predict(p.features), p.label))
val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / validationData.count()


val predictionAndLabel = validationData.map(p => (model.predict(p.features), p.label))
val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / validationData.count()


import org.apache.spark.mllib.evaluation.MulticlassMetrics
 
val metrics = new MulticlassMetrics(predictionAndLabel)
val cfm = metrics.confusionMatrix
 
val tn = cfm(0, 0)
val fp = cfm(0, 1)
val fn = cfm(1, 0)
val tp = cfm(1, 1)


// Confusion matrix
val confusion = metrics.confusionMatrix
val fprINSAT = 100 * (metrics.falsePositiveRate(0) - metrics.falsePositiveRate(0) % 0.0001) 
val fprSAT = 100 * (metrics.falsePositiveRate(1) - metrics.falsePositiveRate(1) % 0.0001) 

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

val metrics2 = new BinaryClassificationMetrics(predictionAndLabel)

// AUPRC
val auPRC = 100 * (metrics2.areaUnderPR() - metrics2.areaUnderPR() % 0.0001) 

// AUROC
val auROC = 100 * (metrics2.areaUnderROC() - metrics2.areaUnderROC() % 0.0001) 

println("Modèle GRADIENT BOOSTING")
printf(s"""|=================== Confusion matrix ==========================
           |_____________| %-15s                     %-15s
           |-------------+-------------------------------------------------
           |Predicted = 0| %-15f                     %-15f
           |Predicted = 1| %-15f                     %-15f
           |===============================================================
         """.stripMargin, "Actual = 0", "Actual = 1", tn, fp, fn, tp)
println()
println(s"Taux de Faux positifs pour les insatisfaits = $fprINSAT %")
println(s"Taux de Faux posisifs pour les satisfaits = $fprSAT %")

println(s"Area under precision-recall curve = $auPRC %")
println(s"Area under ROC = $auROC %")
