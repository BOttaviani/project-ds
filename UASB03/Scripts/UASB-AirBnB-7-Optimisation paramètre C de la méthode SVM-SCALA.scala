
//CONTEXTE DE TRAVAIL
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

//LECTURE D'UN FICHIER AU FORMAT LIBSVM
val echantillon_LIBSVM: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "Data/Vecto_HTF_V6")

//PARTAGE DE L'ECHANTILLON EN JEUX D'APPRENTISSAGE ET DE VALIDATION
val splits = echantillon_LIBSVM.randomSplit(Array(0.7, 0.3))
val (trainingData, validationData) = (splits(0), splits(1))

//APPLICATION DU MODELE AU JEU D'APPRENTISSAGE
val numIterations = 100
val modele_sansC = SVMWithSGD.train(trainingData, numIterations)

//EVALUATION DU MODELE AVEC VALEUR C PAR DEFAUT
modele_sansC.clearThreshold()
// APPLICATION DU MODELE AU JEU DE VALIDATION
val scoreAndLabels_sansC = validationData.map { point =>
  val score = modele_sansC.predict(point.features)
  (score, point.label)
}
// CALCUL DE L'AIRE ROC
val metrics_sansC = new BinaryClassificationMetrics(scoreAndLabels_sansC)
val auROC_sansC = metrics_sansC.areaUnderROC()

//EVALUATION DU MODELE AVEC VALEUR C =0
val C1 = 0.0
val svmAlgC1 = new SVMWithSGD()
svmAlgC1.optimizer
  .setNumIterations(100)
  .setRegParam(C1)
  .setUpdater(new L1Updater)
val modele_avecC1 = svmAlgC1.run(trainingData)

modele_avecC1.clearThreshold()
val scoreAndLabels_avecC1 = validationData.map { point =>
  val score = modele_avecC1.predict(point.features)
  (score, point.label)
}
val metrics_avecC1 = new BinaryClassificationMetrics(scoreAndLabels_avecC1)
val auROC_avecC1 = metrics_avecC1.areaUnderROC()

//EVALUATION DU MODELE AVEC VALEUR C =0.00001
val C2 = 0.00001
val svmAlgC2 = new SVMWithSGD()
svmAlgC2.optimizer
  .setNumIterations(100)
  .setRegParam(C2)
  .setUpdater(new L1Updater)
val modele_avecC2 = svmAlgC2.run(trainingData)

modele_avecC2.clearThreshold()
val scoreAndLabels_avecC2 = validationData.map { point =>
  val score = modele_avecC2.predict(point.features)
  (score, point.label)
}
val metrics_avecC2 = new BinaryClassificationMetrics(scoreAndLabels_avecC2)
val auROC_avecC2 = metrics_avecC2.areaUnderROC()
 
//EVALUATION DU MODELE AVEC VALEUR C =0.0001
val C3 = 0.0001
val svmAlgC3 = new SVMWithSGD()
svmAlgC3.optimizer
  .setNumIterations(100)
  .setRegParam(C3)
  .setUpdater(new L1Updater)
val modele_avecC3 = svmAlgC3.run(trainingData)

modele_avecC3.clearThreshold()
val scoreAndLabels_avecC3 = validationData.map { point =>
  val score = modele_avecC3.predict(point.features)
  (score, point.label)
}
val metrics_avecC3 = new BinaryClassificationMetrics(scoreAndLabels_avecC3)
val auROC_avecC3 = metrics_avecC3.areaUnderROC()

//EVALUATION DU MODELE AVEC VALEUR C =0.001
val C4 = 0.001
val svmAlgC4 = new SVMWithSGD()
svmAlgC4.optimizer
  .setNumIterations(100)
  .setRegParam(C4)
  .setUpdater(new L1Updater)
val modele_avecC4 = svmAlgC4.run(trainingData)
modele_avecC4.clearThreshold()

val scoreAndLabels_avecC4 = validationData.map { point =>
  val score = modele_avecC4.predict(point.features)
  (score, point.label)
}
val metrics_avecC4 = new BinaryClassificationMetrics(scoreAndLabels_avecC4)
val auROC_avecC4 = metrics_avecC4.areaUnderROC()

//EVALUATION DU MODELE AVEC VALEUR C =0.01
val C5 = 0.01
val svmAlgC5 = new SVMWithSGD()
svmAlgC5.optimizer
  .setNumIterations(100)
  .setRegParam(C5)
  .setUpdater(new L1Updater)
val modele_avecC5 = svmAlgC5.run(trainingData)
modele_avecC5.clearThreshold()
val scoreAndLabels_avecC5 = validationData.map { point =>
  val score = modele_avecC5.predict(point.features)
  (score, point.label)
}
val metrics_avecC5 = new BinaryClassificationMetrics(scoreAndLabels_avecC5)
val auROC_avecC5 = metrics_avecC5.areaUnderROC()

//EVALUATION DU MODELE AVEC VALEUR C =0.1
val C6 = 0.1
val svmAlgC6 = new SVMWithSGD()
svmAlgC6.optimizer
  .setNumIterations(100)
  .setRegParam(C6)
  .setUpdater(new L1Updater)
val modele_avecC6 = svmAlgC6.run(trainingData)
modele_avecC6.clearThreshold()
val scoreAndLabels_avecC6 = validationData.map { point =>
  val score = modele_avecC6.predict(point.features)
  (score, point.label)
}
val metrics_avecC6 = new BinaryClassificationMetrics(scoreAndLabels_avecC6)
val auROC_avecC6 = metrics_avecC6.areaUnderROC()

//EVALUATION DU MODELE AVEC VALEUR C =0.5
val C7 = 0.5
val svmAlgC7 = new SVMWithSGD()
svmAlgC7.optimizer
  .setNumIterations(100)
  .setRegParam(C7)
  .setUpdater(new L1Updater)
val modele_avecC7 = svmAlgC7.run(trainingData)
modele_avecC7.clearThreshold()
val scoreAndLabels_avecC7 = validationData.map { point =>
  val score = modele_avecC7.predict(point.features)
  (score, point.label)
}
val metrics_avecC7 = new BinaryClassificationMetrics(scoreAndLabels_avecC7)
val auROC_avecC7 = metrics_avecC7.areaUnderROC()


println("Vectorisation avec HashingTF")
println("==========================================================")
println(s"Area under ROC pour methode sans Regparam = $auROC_sansC")
println(s"paramètre de régularisation : C = $C1")
println(s"Area under ROC pour methode avec Regparam = $auROC_avecC1")
println(s"paramètre de régularisation : C = $C2")
println(s"Area under ROC pour methode avec Regparam = $auROC_avecC2")
println(s"paramètre de régularisation : C = $C3")
println(s"Area under ROC pour methode avec Regparam = $auROC_avecC3")
println(s"paramètre de régularisation : C = $C4")
println(s"Area under ROC pour methode avec Regparam = $auROC_avecC4")
println(s"paramètre de régularisation : C = $C5")
println(s"Area under ROC pour methode avec Regparam = $auROC_avecC5")
println(s"paramètre de régularisation : C = $C6")
println(s"Area under ROC pour methode avec Regparam = $auROC_avecC6")
println(s"paramètre de régularisation : C = $C7")
println(s"Area under ROC pour methode avec Regparam = $auROC_avecC7")

//LECTURE D'UN FICHIER AU FORMAT LIBSVM
val echantillon_LIBSVM: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "Data/Vecto_Word2vec_V6")

//PARTAGE DE L'ECHANTILLON EN JEUX D'APPRENTISSAGE ET DE VALIDATION
val splits = echantillon_LIBSVM.randomSplit(Array(0.7, 0.3))
val (trainingData, validationData) = (splits(0), splits(1))

//APPLICATION DU MODELE AU JEU D'APPRENTISSAGE
val numIterations = 100
val modele_sansC = SVMWithSGD.train(trainingData, numIterations)

//EVALUATION DU MODELE AVEC VALEUR C PAR DEFAUT
modele_sansC.clearThreshold()
//APPLICATION DU MODELE AU JEU DE VALIDATION
val scoreAndLabels_sansC = validationData.map { point =>
  val score = modele_sansC.predict(point.features)
  (score, point.label)
}
// CALCUL DE L'AIRE ROC
val metrics_sansC = new BinaryClassificationMetrics(scoreAndLabels_sansC)
val auROC_sansC = metrics_sansC.areaUnderROC()

//EVALUATION DU MODELE AVEC VALEUR C = 0.0
val C1 = 0.0
val svmAlgC1 = new SVMWithSGD()
svmAlgC1.optimizer
  .setNumIterations(100)
  .setRegParam(C1)
  .setUpdater(new L1Updater)
val modele_avecC1 = svmAlgC1.run(trainingData)
modele_avecC1.clearThreshold()

val scoreAndLabels_avecC1 = validationData.map { point =>
  val score = modele_avecC1.predict(point.features)
  (score, point.label)
}
val metrics_avecC1 = new BinaryClassificationMetrics(scoreAndLabels_avecC1)
val auROC_avecC1 = metrics_avecC1.areaUnderROC()

//EVALUATION DU MODELE AVEC VALEUR C = 0.00001
val C2 = 0.00001
val svmAlgC2 = new SVMWithSGD()
svmAlgC2.optimizer
  .setNumIterations(100)
  .setRegParam(C2)
  .setUpdater(new L1Updater)
val modele_avecC2 = svmAlgC2.run(trainingData)

modele_avecC2.clearThreshold()
val scoreAndLabels_avecC2 = validationData.map { point =>
  val score = modele_avecC2.predict(point.features)
  (score, point.label)
}
val metrics_avecC2 = new BinaryClassificationMetrics(scoreAndLabels_avecC2)
val auROC_avecC2 = metrics_avecC2.areaUnderROC()
 
//EVALUATION DU MODELE AVEC VALEUR C = 0.0001
val C3 = 0.0001
val svmAlgC3 = new SVMWithSGD()
svmAlgC3.optimizer
  .setNumIterations(100)
  .setRegParam(C3)
  .setUpdater(new L1Updater)
val modele_avecC3 = svmAlgC3.run(trainingData)
modele_avecC3.clearThreshold()
val scoreAndLabels_avecC3 = validationData.map { point =>
  val score = modele_avecC3.predict(point.features)
  (score, point.label)
}
val metrics_avecC3 = new BinaryClassificationMetrics(scoreAndLabels_avecC3)
val auROC_avecC3 = metrics_avecC3.areaUnderROC()

//EVALUATION DU MODELE AVEC VALEUR C = 0.0001
val C4 = 0.001
val svmAlgC4 = new SVMWithSGD()
svmAlgC4.optimizer
  .setNumIterations(100)
  .setRegParam(C4)
  .setUpdater(new L1Updater)
val modele_avecC4 = svmAlgC4.run(trainingData)

modele_avecC4.clearThreshold()
val scoreAndLabels_avecC4 = validationData.map { point =>
  val score = modele_avecC4.predict(point.features)
  (score, point.label)
}
val metrics_avecC4 = new BinaryClassificationMetrics(scoreAndLabels_avecC4)
val auROC_avecC4 = metrics_avecC4.areaUnderROC()

//EVALUATION DU MODELE AVEC VALEUR C = 0.01
val C5 = 0.01
val svmAlgC5 = new SVMWithSGD()
svmAlgC5.optimizer
  .setNumIterations(100)
  .setRegParam(C5)
  .setUpdater(new L1Updater)
val modele_avecC5 = svmAlgC5.run(trainingData)

modele_avecC5.clearThreshold()
val scoreAndLabels_avecC5 = validationData.map { point =>
  val score = modele_avecC5.predict(point.features)
  (score, point.label)
}
val metrics_avecC5 = new BinaryClassificationMetrics(scoreAndLabels_avecC5)
val auROC_avecC5 = metrics_avecC5.areaUnderROC()
println(s"paramètre de régularisation : C = $C5")
println(s"Area under ROC pour methode avec Regparam = $auROC_avecC5")

//EVALUATION DU MODELE AVEC VALEUR C = 0.01
val C6 = 0.1
val svmAlgC6 = new SVMWithSGD()
svmAlgC6.optimizer
  .setNumIterations(100)
  .setRegParam(C6)
  .setUpdater(new L1Updater)
val modele_avecC6 = svmAlgC6.run(trainingData)

modele_avecC6.clearThreshold()
val scoreAndLabels_avecC6 = validationData.map { point =>
  val score = modele_avecC6.predict(point.features)
  (score, point.label)
}
val metrics_avecC6 = new BinaryClassificationMetrics(scoreAndLabels_avecC6)
val auROC_avecC6 = metrics_avecC6.areaUnderROC()

//EVALUATION DU MODELE AVEC VALEUR C = 0.5
val C7 = 0.5
val svmAlgC7 = new SVMWithSGD()
svmAlgC7.optimizer
  .setNumIterations(100)
  .setRegParam(C7)
  .setUpdater(new L1Updater)
val modele_avecC7 = svmAlgC7.run(trainingData)

modele_avecC7.clearThreshold()
val scoreAndLabels_avecC7 = validationData.map { point =>
  val score = modele_avecC7.predict(point.features)
  (score, point.label)
}
val metrics_avecC7 = new BinaryClassificationMetrics(scoreAndLabels_avecC7)
val auROC_avecC7 = metrics_avecC7.areaUnderROC()


println("Vectorisation avec Word2Vec Corpus1")
println("==========================================================")
println(s"Area under ROC pour methode sans Regparam = $auROC_sansC")
println(s"paramètre de régularisation : C = $C1")
println(s"Area under ROC pour methode avec Regparam = $auROC_avecC1")
println(s"paramètre de régularisation : C = $C2")
println(s"Area under ROC pour methode avec Regparam = $auROC_avecC2")
println(s"paramètre de régularisation : C = $C3")
println(s"Area under ROC pour methode avec Regparam = $auROC_avecC3")
println(s"paramètre de régularisation : C = $C4")
println(s"Area under ROC pour methode avec Regparam = $auROC_avecC4")
println(s"paramètre de régularisation : C = $C5")
println(s"Area under ROC pour methode avec Regparam = $auROC_avecC5")
println(s"paramètre de régularisation : C = $C6")
println(s"Area under ROC pour methode avec Regparam = $auROC_avecC6")
println(s"paramètre de régularisation : C = $C7")
println(s"Area under ROC pour methode avec Regparam = $auROC_avecC7")

//LECTURE D'UN FICHIER AU FORMAT LIBSVM
val echantillon_LIBSVM: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "Data/Vecto_Word2vecC2_V6")

//PARTAGE DE L'ECHANTILLON EN JEUX D'APPRENTISSAGE ET DE VALIDATION
val splits = echantillon_LIBSVM.randomSplit(Array(0.7, 0.3))
val (trainingData, validationData) = (splits(0), splits(1))

//APPLICATION DU MODELE AU JEU D'APPRENTISSAGE
val numIterations = 100
val modele_sansC = SVMWithSGD.train(trainingData, numIterations)

//EVALUATION DU MODELE AVEC C PAR DEFAUT
modele_sansC.clearThreshold()
//APPLICATION DU MODELE AU JEU DE VALIDATION
val scoreAndLabels_sansC = validationData.map { point =>
  val score = modele_sansC.predict(point.features)
  (score, point.label)
}
// CALCUL DE L'AIRE ROC
val metrics_sansC = new BinaryClassificationMetrics(scoreAndLabels_sansC)
val auROC_sansC = metrics_sansC.areaUnderROC()

//EVALUATION DU MODELE AVEC VALEUR C = 0.0
val C1 = 0.0
val svmAlgC1 = new SVMWithSGD()
svmAlgC1.optimizer
  .setNumIterations(100)
  .setRegParam(C1)
  .setUpdater(new L1Updater)
val modele_avecC1 = svmAlgC1.run(trainingData)

modele_avecC1.clearThreshold()
val scoreAndLabels_avecC1 = validationData.map { point =>
  val score = modele_avecC1.predict(point.features)
  (score, point.label)
}
val metrics_avecC1 = new BinaryClassificationMetrics(scoreAndLabels_avecC1)
val auROC_avecC1 = metrics_avecC1.areaUnderROC()

//EVALUATION DU MODELE AVEC VALEUR C = 0.00001
val C2 = 0.00001
val svmAlgC2 = new SVMWithSGD()
svmAlgC2.optimizer
  .setNumIterations(100)
  .setRegParam(C2)
  .setUpdater(new L1Updater)
val modele_avecC2 = svmAlgC2.run(trainingData)

modele_avecC2.clearThreshold()
val scoreAndLabels_avecC2 = validationData.map { point =>
  val score = modele_avecC2.predict(point.features)
  (score, point.label)
}
val metrics_avecC2 = new BinaryClassificationMetrics(scoreAndLabels_avecC2)
val auROC_avecC2 = metrics_avecC2.areaUnderROC()
 
//EVALUATION DU MODELE AVEC VALEUR C = 0.0001
val C3 = 0.0001
val svmAlgC3 = new SVMWithSGD()
svmAlgC3.optimizer
  .setNumIterations(100)
  .setRegParam(C3)
  .setUpdater(new L1Updater)
val modele_avecC3 = svmAlgC3.run(trainingData)

modele_avecC3.clearThreshold()
val scoreAndLabels_avecC3 = validationData.map { point =>
  val score = modele_avecC3.predict(point.features)
  (score, point.label)
}
val metrics_avecC3 = new BinaryClassificationMetrics(scoreAndLabels_avecC3)
val auROC_avecC3 = metrics_avecC3.areaUnderROC()

//EVALUATION DU MODELE AVEC VALEUR C = 0.001
val C4 = 0.001
val svmAlgC4 = new SVMWithSGD()
svmAlgC4.optimizer
  .setNumIterations(100)
  .setRegParam(C4)
  .setUpdater(new L1Updater)
val modele_avecC4 = svmAlgC4.run(trainingData)

modele_avecC4.clearThreshold()
val scoreAndLabels_avecC4 = validationData.map { point =>
  val score = modele_avecC4.predict(point.features)
  (score, point.label)
}
val metrics_avecC4 = new BinaryClassificationMetrics(scoreAndLabels_avecC4)
val auROC_avecC4 = metrics_avecC4.areaUnderROC()

//EVALUATION DU MODELE AVEC VALEUR C = 0.01
val C5 = 0.01
val svmAlgC5 = new SVMWithSGD()
svmAlgC5.optimizer
  .setNumIterations(100)
  .setRegParam(C5)
  .setUpdater(new L1Updater)
val modele_avecC5 = svmAlgC5.run(trainingData)

modele_avecC5.clearThreshold()
val scoreAndLabels_avecC5 = validationData.map { point =>
  val score = modele_avecC5.predict(point.features)
  (score, point.label)
}
val metrics_avecC5 = new BinaryClassificationMetrics(scoreAndLabels_avecC5)
val auROC_avecC5 = metrics_avecC5.areaUnderROC()

//EVALUATION DU MODELE AVEC VALEUR C = 0.1
val C6 = 0.1
val svmAlgC6 = new SVMWithSGD()
svmAlgC6.optimizer
  .setNumIterations(100)
  .setRegParam(C6)
  .setUpdater(new L1Updater)
val modele_avecC6 = svmAlgC6.run(trainingData)

modele_avecC6.clearThreshold()
val scoreAndLabels_avecC6 = validationData.map { point =>
  val score = modele_avecC6.predict(point.features)
  (score, point.label)
}
val metrics_avecC6 = new BinaryClassificationMetrics(scoreAndLabels_avecC6)
val auROC_avecC6 = metrics_avecC6.areaUnderROC()

//EVALUATION DU MODELE AVEC VALEUR C = 0.5
val C7 = 0.5
val svmAlgC7 = new SVMWithSGD()
svmAlgC7.optimizer
  .setNumIterations(100)
  .setRegParam(C7)
  .setUpdater(new L1Updater)
val modele_avecC7 = svmAlgC7.run(trainingData)

modele_avecC7.clearThreshold()
val scoreAndLabels_avecC7 = validationData.map { point =>
  val score = modele_avecC7.predict(point.features)
  (score, point.label)
}
val metrics_avecC7 = new BinaryClassificationMetrics(scoreAndLabels_avecC7)
val auROC_avecC7 = metrics_avecC7.areaUnderROC()


println("Vectorisation avec Word2Vec Corpus2")
println("==========================================================")
println(s"Area under ROC pour methode sans Regparam = $auROC_sansC")
println(s"paramètre de régularisation : C = $C1")
println(s"Area under ROC pour methode avec Regparam = $auROC_avecC1")
println(s"paramètre de régularisation : C = $C2")
println(s"Area under ROC pour methode avec Regparam = $auROC_avecC2")
println(s"paramètre de régularisation : C = $C3")
println(s"Area under ROC pour methode avec Regparam = $auROC_avecC3")
println(s"paramètre de régularisation : C = $C4")
println(s"Area under ROC pour methode avec Regparam = $auROC_avecC4")
println(s"paramètre de régularisation : C = $C5")
println(s"Area under ROC pour methode avec Regparam = $auROC_avecC5")
println(s"paramètre de régularisation : C = $C6")
println(s"Area under ROC pour methode avec Regparam = $auROC_avecC6")
println(s"paramètre de régularisation : C = $C7")
println(s"Area under ROC pour methode avec Regparam = $auROC_avecC7")

//LECTURE D'UN FICHIER AU FORMAT LIBSVM
val echantillon_LIBSVM: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "Data/Vecto_Countvectorizer_V6")

//PARTAGE DE L'ECHANTILLON EN JEUX D'APPRENTISSAGE ET DE VALIDATION
val splits = echantillon_LIBSVM.randomSplit(Array(0.7, 0.3))
val (trainingData, validationData) = (splits(0), splits(1))

//APPLICATION DU MODELE AU JEU D'APPRENTISSAGE
val numIterations = 100
val modele_sansC = SVMWithSGD.train(trainingData, numIterations)

//EVALUATION DU MODELE AEC VALEUR C PAR DEFAUT
modele_sansC.clearThreshold()
//APPLICATION DU MODELE AU JEU DE VALIDATION
val scoreAndLabels_sansC = validationData.map { point =>
  val score = modele_sansC.predict(point.features)
  (score, point.label)
}
// CALCUL DE L'AIRE ROC
val metrics_sansC = new BinaryClassificationMetrics(scoreAndLabels_sansC)
val auROC_sansC = metrics_sansC.areaUnderROC()

//EVALUATION DU MODELE AVEC VALEUR C = 0.0
val C1 = 0.0
val svmAlgC1 = new SVMWithSGD()
svmAlgC1.optimizer
  .setNumIterations(100)
  .setRegParam(C1)
  .setUpdater(new L1Updater)
val modele_avecC1 = svmAlgC1.run(trainingData)

modele_avecC1.clearThreshold()
val scoreAndLabels_avecC1 = validationData.map { point =>
  val score = modele_avecC1.predict(point.features)
  (score, point.label)
}
val metrics_avecC1 = new BinaryClassificationMetrics(scoreAndLabels_avecC1)
val auROC_avecC1 = metrics_avecC1.areaUnderROC()

//EVALUATION DU MODELE AVEC VALEUR C = 0.00001
val C2 = 0.00001
val svmAlgC2 = new SVMWithSGD()
svmAlgC2.optimizer
  .setNumIterations(100)
  .setRegParam(C2)
  .setUpdater(new L1Updater)
val modele_avecC2 = svmAlgC2.run(trainingData)

modele_avecC2.clearThreshold()
val scoreAndLabels_avecC2 = validationData.map { point =>
  val score = modele_avecC2.predict(point.features)
  (score, point.label)
}
val metrics_avecC2 = new BinaryClassificationMetrics(scoreAndLabels_avecC2)
val auROC_avecC2 = metrics_avecC2.areaUnderROC()
 
//EVALUATION DU MODELE AVEC VALEUR C = 0.0001
val C3 = 0.0001
val svmAlgC3 = new SVMWithSGD()
svmAlgC3.optimizer
  .setNumIterations(100)
  .setRegParam(C3)
  .setUpdater(new L1Updater)
val modele_avecC3 = svmAlgC3.run(trainingData)

modele_avecC3.clearThreshold()
val scoreAndLabels_avecC3 = validationData.map { point =>
  val score = modele_avecC3.predict(point.features)
  (score, point.label)
}
val metrics_avecC3 = new BinaryClassificationMetrics(scoreAndLabels_avecC3)
val auROC_avecC3 = metrics_avecC3.areaUnderROC()

//EVALUATION DU MODELE AVEC VALEUR C = 0.001
val C4 = 0.001
val svmAlgC4 = new SVMWithSGD()
svmAlgC4.optimizer
  .setNumIterations(100)
  .setRegParam(C4)
  .setUpdater(new L1Updater)
val modele_avecC4 = svmAlgC4.run(trainingData)

modele_avecC4.clearThreshold()
val scoreAndLabels_avecC4 = validationData.map { point =>
  val score = modele_avecC4.predict(point.features)
  (score, point.label)
}
val metrics_avecC4 = new BinaryClassificationMetrics(scoreAndLabels_avecC4)
val auROC_avecC4 = metrics_avecC4.areaUnderROC()

//EVALUATION DU MODELE AVEC VALEUR C = 0.01
val C5 = 0.01
val svmAlgC5 = new SVMWithSGD()
svmAlgC5.optimizer
  .setNumIterations(100)
  .setRegParam(C5)
  .setUpdater(new L1Updater)
val modele_avecC5 = svmAlgC5.run(trainingData)

modele_avecC5.clearThreshold()
val scoreAndLabels_avecC5 = validationData.map { point =>
  val score = modele_avecC5.predict(point.features)
  (score, point.label)
}
val metrics_avecC5 = new BinaryClassificationMetrics(scoreAndLabels_avecC5)
val auROC_avecC5 = metrics_avecC5.areaUnderROC()

//EVALUATION DU MODELE AVEC VALEUR C = 0.1
val C6 = 0.1
val svmAlgC6 = new SVMWithSGD()
svmAlgC6.optimizer
  .setNumIterations(100)
  .setRegParam(C6)
  .setUpdater(new L1Updater)
val modele_avecC6 = svmAlgC6.run(trainingData)

modele_avecC6.clearThreshold()
val scoreAndLabels_avecC6 = validationData.map { point =>
  val score = modele_avecC6.predict(point.features)
  (score, point.label)
}
val metrics_avecC6 = new BinaryClassificationMetrics(scoreAndLabels_avecC6)
val auROC_avecC6 = metrics_avecC6.areaUnderROC()

//EVALUATION DU MODELE AVEC VALEUR C = 0.5
val C7 = 0.5
val svmAlgC7 = new SVMWithSGD()
svmAlgC7.optimizer
  .setNumIterations(100)
  .setRegParam(C7)
  .setUpdater(new L1Updater)
val modele_avecC7 = svmAlgC7.run(trainingData)

modele_avecC7.clearThreshold()
val scoreAndLabels_avecC7 = validationData.map { point =>
  val score = modele_avecC7.predict(point.features)
  (score, point.label)
}
val metrics_avecC7 = new BinaryClassificationMetrics(scoreAndLabels_avecC7)
val auROC_avecC7 = metrics_avecC7.areaUnderROC()


println("Vectorisation avec CountVectorizer")
println("==========================================================")
println(s"Area under ROC pour methode sans Regparam = $auROC_sansC")
println(s"paramètre de régularisation : C = $C1")
println(s"Area under ROC pour methode avec Regparam = $auROC_avecC1")
println(s"paramètre de régularisation : C = $C2")
println(s"Area under ROC pour methode avec Regparam = $auROC_avecC2")
println(s"paramètre de régularisation : C = $C3")
println(s"Area under ROC pour methode avec Regparam = $auROC_avecC3")
println(s"paramètre de régularisation : C = $C4")
println(s"Area under ROC pour methode avec Regparam = $auROC_avecC4")
println(s"paramètre de régularisation : C = $C5")
println(s"Area under ROC pour methode avec Regparam = $auROC_avecC5")
println(s"paramètre de régularisation : C = $C6")
println(s"Area under ROC pour methode avec Regparam = $auROC_avecC6")
println(s"paramètre de régularisation : C = $C7")
println(s"Area under ROC pour methode avec Regparam = $auROC_avecC7")
