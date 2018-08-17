
//CONTEXTE DE TRAVAIL
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

//LECTURE D'UN FICHIER AU FORMAT LIBSVM
val echantillon_LIBSVM: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "Data/Vecto_HTF")

//PARTAGE DE L'ECHANTILLON EN JEUX D'APPRENTISSAGE ET DE VALIDATION
val splits = echantillon_LIBSVM.randomSplit(Array(0.7, 0.3))
val (trainingData, validationData) = (splits(0), splits(1))

//CONSTRUCTION DU MODELE
val boostingStrategy = BoostingStrategy.defaultParams("Classification")
boostingStrategy.setNumIterations(20) //number of passes over our training data
boostingStrategy.treeStrategy.setNumClasses(2) //We have two output classes: happy and sad
boostingStrategy.treeStrategy.setMaxDepth(5)

//APPLICATION DU MODELE AU JEU D'APPRENTISSAGE
val modele = GradientBoostedTrees.train(trainingData, boostingStrategy)
// Sauvegarde du modèle
modele.save(sc, "modele/GBT_HTF")

//EVALUATION DU MODELE
//pour le jeu de validation
var labelAndPredsValid = validationData.map { point =>
  val prediction = modele.predict(point.features)
  Tuple2(point.label, prediction)
}


// CALCUL DES INDICATEURS DE PERFORMANCE
val metrics_GBT_HTF = new BinaryClassificationMetrics(labelAndPredsValid)
val auPRC_GBT_HTF = 100 * (metrics_GBT_HTF.areaUnderPR() - metrics_GBT_HTF.areaUnderPR() % 0.0001) 
val auROC_GBT_HTF = 100 * (metrics_GBT_HTF.areaUnderROC() - metrics_GBT_HTF.areaUnderROC() % 0.0001) 

//LECTURE D'UN FICHIER AU FORMAT LIBSVM
val echantillon_LIBSVM: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "Data/Vecto_Word2vec")

//PARTAGE DE L'ECHANTILLON EN JEUX D'APPRENTISSAGE ET DE VALIDATION
val splits = echantillon_LIBSVM.randomSplit(Array(0.7, 0.3))
val (trainingData, validationData) = (splits(0), splits(1))

//CONSTRUCTION DU MODELE

val boostingStrategy = BoostingStrategy.defaultParams("Classification")
boostingStrategy.setNumIterations(20) //number of passes over our training data
boostingStrategy.treeStrategy.setNumClasses(2) //We have two output classes: happy and sad
boostingStrategy.treeStrategy.setMaxDepth(5)

//APPLICATION DU MODELE AU JEU D'APPRENTISSAGE
val modele = GradientBoostedTrees.train(trainingData, boostingStrategy)
// Sauvegarde du modèle
modele.save(sc, "modele/GBT_W2V")

//EVALUATION DU MODELE

//pour le jeu de validation
var labelAndPredsValid = validationData.map { point =>
  val prediction = modele.predict(point.features)
  Tuple2(point.label, prediction)
}


//CALCUL DES INDICATEURS DE PERFORMANCE
val metrics_GBT_W2V = new BinaryClassificationMetrics(labelAndPredsValid)
val auPRC_GBT_W2V = 100 * (metrics_GBT_W2V.areaUnderPR() - metrics_GBT_W2V.areaUnderPR() % 0.0001) 
val auROC_GBT_W2V = 100 * (metrics_GBT_W2V.areaUnderROC() - metrics_GBT_W2V.areaUnderROC() % 0.0001) 

//LECTURE D'UN FICHIER AU FORMAT LIBSVM
val echantillon_LIBSVM: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "Data/Vecto_Word2vecC2")

//PARTAGE DE L'ECHANTILLON EN JEUX D'APPRENTISSAGE ET DE VALIDATION
val splits = echantillon_LIBSVM.randomSplit(Array(0.7, 0.3))
val (trainingData, validationData) = (splits(0), splits(1))

//CONSTRUCTION DU MODELE

val boostingStrategy = BoostingStrategy.defaultParams("Classification")
boostingStrategy.setNumIterations(20) //number of passes over our training data
boostingStrategy.treeStrategy.setNumClasses(2) //We have two output classes: happy and sad
boostingStrategy.treeStrategy.setMaxDepth(5)

//APPLICATION DU MODELE AU JEU D'APPRENTISSAGE
val modele = GradientBoostedTrees.train(trainingData, boostingStrategy)
// Sauvegarde du modèle
modele.save(sc, "modele/GBT_W2VC2")

//EVALUATION DU MODELE

//pour le jeu de validation
var labelAndPredsValid = validationData.map { point =>
  val prediction = modele.predict(point.features)
  Tuple2(point.label, prediction)
}

//CALCUL DES INDICATEURS DE PERFORMANCE
val metrics_GBT_W2V_C2 = new BinaryClassificationMetrics(labelAndPredsValid)
val auPRC_GBT_W2V_C2 = 100 * (metrics_GBT_W2V_C2.areaUnderPR() - metrics_GBT_W2V_C2.areaUnderPR() % 0.0001) 
val auROC_GBT_W2V_C2 = 100 * (metrics_GBT_W2V_C2.areaUnderROC() - metrics_GBT_W2V_C2.areaUnderROC() % 0.0001) 

//LECTURE D'UN FICHIER AU FORMAT LIBSVM
val echantillon_LIBSVM: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "Data/Vecto_Countvectorizer")

//PARTAGE DE L'ECHANTILLON EN JEUX D'APPRENTISSAGE ET DE VALIDATION
val splits = echantillon_LIBSVM.randomSplit(Array(0.7, 0.3))
val (trainingData, validationData) = (splits(0), splits(1))

//CONSTRUCTION DU MODELE

val boostingStrategy = BoostingStrategy.defaultParams("Classification")
boostingStrategy.setNumIterations(20) //number of passes over our training data
boostingStrategy.treeStrategy.setNumClasses(2) //We have two output classes: happy and sad
boostingStrategy.treeStrategy.setMaxDepth(5)

//APPLICATION DU MODELE AU JEU D'APPRENTISSAGE
val modele = GradientBoostedTrees.train(trainingData, boostingStrategy)
// Sauvegarde du modèle
modele.save(sc, "modele/GBT_CV")

//EVALUATION DU MODELE
//pour le jeu de validation
var labelAndPredsValid = validationData.map { point =>
  val prediction = modele.predict(point.features)
  Tuple2(point.label, prediction)
}


//CALCUL DES INDICATEURS DE PERFORMANCE
val metrics_GBT_CV = new BinaryClassificationMetrics(labelAndPredsValid)
val auPRC_GBT_CV = 100 * (metrics_GBT_CV.areaUnderPR() - metrics_GBT_CV.areaUnderPR() % 0.0001) 
val auROC_GBT_CV = 100 * (metrics_GBT_CV.areaUnderROC() - metrics_GBT_CV.areaUnderROC() % 0.0001) 

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

//LECTURE D'UN FICHIER AU FORMAT LIBSVM
val echantillon_LIBSVM: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "Data/Vecto_HTF")

//PARTAGE DE L'ECHANTILLON EN JEUX D'APPRENTISSAGE ET DE VALIDATION
val splits = echantillon_LIBSVM.randomSplit(Array(0.7, 0.3))
val (trainingData, validationData) = (splits(0), splits(1))

//APPLICATION DU MODELE AU JEU D'APPRENTISSAGE
val numIterations = 100
val model = SVMWithSGD.train(trainingData, numIterations)
// Sauvegarde du modèle
model.save(sc, "modele/SVM_HTF")

//EVALUATION DU MODELE
// Clear the default threshold.
model.clearThreshold()
// Prédiction sur le jeu de validation
val predictionAndLabels = validationData.map { case LabeledPoint(label, features) =>
  val prediction = model.predict(features)
  (prediction, label)
}


// CALCUL DES INDICATEURS DE PERFORMANCE
val metrics_SVM_HTF = new BinaryClassificationMetrics(predictionAndLabels)
val auPRC_SVM_HTF = 100 * (metrics_SVM_HTF.areaUnderPR() - metrics_SVM_HTF.areaUnderPR() % 0.0001) 
val auROC_SVM_HTF = 100 * (metrics_SVM_HTF.areaUnderROC() - metrics_SVM_HTF.areaUnderROC() % 0.0001) 

//LECTURE D'UN FICHIER AU FORMAT LIBSVM
val echantillon_LIBSVM: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "Data/Vecto_Word2vec")

//PARTAGE DE L'ECHANTILLON EN JEUX D'APPRENTISSAGE ET DE VALIDATION
val splits = echantillon_LIBSVM.randomSplit(Array(0.7, 0.3))
val (trainingData, validationData) = (splits(0), splits(1))

//APPLICATION DU MODELE AU JEU D'APPRENTISSAGE
val numIterations = 100
val model = SVMWithSGD.train(trainingData, numIterations)
// Sauvegarde du modèle
model.save(sc, "modele/SVM_W2V")
model.clearThreshold()
// Prédiction sur le jeu de validation
val predictionAndLabels = validationData.map { case LabeledPoint(label, features) =>
  val prediction = model.predict(features)
  (prediction, label)
}


//CALCUL DES INDICATEURS DE PERFORMANCE
val metrics_SVM_W2V = new BinaryClassificationMetrics(predictionAndLabels)
val auPRC_SVM_W2V = 100 * (metrics_SVM_W2V.areaUnderPR() - metrics_SVM_W2V.areaUnderPR() % 0.0001) 
val auROC_SVM_W2V = 100 * (metrics_SVM_W2V.areaUnderROC() - metrics_SVM_W2V.areaUnderROC() % 0.0001) 

//LECTURE D'UN FICHIER AU FORMAT LIBSVM
val echantillon_LIBSVM: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "Data/Vecto_Word2vecC2")

//PARTAGE DE L'ECHANTILLON EN JEUX D'APPRENTISSAGE ET DE VALIDATION
val splits = echantillon_LIBSVM.randomSplit(Array(0.7, 0.3))
val (trainingData, validationData) = (splits(0), splits(1))

//APPLICATION DU MODELE AU JEU D'APPRENTISSAGE
val numIterations = 100
val model = SVMWithSGD.train(trainingData, numIterations)
// Sauvegarde du modèle
model.save(sc, "modele/SVM_W2VC2")

//EVALUATION DU MODELE
// Clear the default threshold.
model.clearThreshold()
// Prédiction sur le jeu de validation
val predictionAndLabels = validationData.map { case LabeledPoint(label, features) =>
  val prediction = model.predict(features)
  (prediction, label)
}


// CALCUL DES INDICATEURS DE PERFORMANCE
val metrics_SVM_W2V_C2 = new BinaryClassificationMetrics(predictionAndLabels)
val auPRC_SVM_W2V_C2 = 100 * (metrics_SVM_W2V_C2.areaUnderPR() - metrics_SVM_W2V_C2.areaUnderPR() % 0.0001) 
val auROC_SVM_W2V_C2 = 100 * (metrics_SVM_W2V_C2.areaUnderROC() - metrics_SVM_W2V_C2.areaUnderROC() % 0.0001) 

//LECTURE D'UN FICHIER AU FORMAT LIBSVM
val echantillon_LIBSVM: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "Data/Vecto_Countvectorizer")

//PARTAGE DE L'ECHANTILLON EN JEUX D'APPRENTISSAGE ET DE VALIDATION
val splits = echantillon_LIBSVM.randomSplit(Array(0.7, 0.3))
val (trainingData, validationData) = (splits(0), splits(1))

//APPLICATION DU MODELE AU JEU D'APPRENTISSAGE
val numIterations = 100
val model = SVMWithSGD.train(trainingData, numIterations)

// Sauvegarde du modèle
model.save(sc, "modele/SVM_CV")

//EVALUATION DU MODELE
// Clear the default threshold.
model.clearThreshold()
// Prédiction sur le jeu de validation
val predictionAndLabels = validationData.map { case LabeledPoint(label, features) =>
  val prediction = model.predict(features)
  (prediction, label)
}

// CALCUL DES INDICATEURS DE PERFORMANCE
val metrics_SVM_CV = new BinaryClassificationMetrics(predictionAndLabels)
val auPRC_SVM_CV = 100 * (metrics_SVM_CV.areaUnderPR() - metrics_SVM_CV.areaUnderPR() % 0.0001) 
val auROC_SVM_CV = 100 * (metrics_SVM_CV.areaUnderROC() - metrics_SVM_CV.areaUnderROC() % 0.0001) 

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

//LECTURE D'UN FICHIER AU FORMAT LIBSVM
val echantillon_LIBSVM: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "Data/Vecto_HTF")

//PARTAGE DE L'ECHANTILLON EN JEUX D'APPRENTISSAGE ET DE VALIDATION
val splits = echantillon_LIBSVM.randomSplit(Array(0.7, 0.3))
val (trainingData, validationData) = (splits(0), splits(1))

//CONSTRUCTION ET APPLICATION DU MODELE AU JEU D'APPRENTISSAGE
val model = new LogisticRegressionWithLBFGS()
  .setNumClasses(2)
  .run(trainingData)

// Sauvegarde du modèle
model.save(sc, "modele/LGB_HTF")

//APPLICATION DU MODELE AU JEU DE VALIDATION
val predictionAndLabels = validationData.map { case LabeledPoint(label, features) =>
  val prediction = model.predict(features)
  (prediction, label)
}

//CALCUL DES INDICATEURS DE PERFORMANCE
val metrics_LGB_HTF = new BinaryClassificationMetrics(predictionAndLabels)
val auPRC_LGB_HTF = 100 * (metrics_LGB_HTF.areaUnderPR() - metrics_LGB_HTF.areaUnderPR() % 0.0001) 
val auROC_LGB_HTF = 100 * (metrics_LGB_HTF.areaUnderROC() - metrics_LGB_HTF.areaUnderROC() % 0.0001) 

//LECTURE D'UN FICHIER AU FORMAT LIBSVM
val echantillon_LIBSVM: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "Data/Vecto_Word2vec")

//PARTAGE DE L'ECHANTILLON EN JEUX D'APPRENTISSAGE ET DE VALIDATION
val splits = echantillon_LIBSVM.randomSplit(Array(0.7, 0.3))
val (trainingData, validationData) = (splits(0), splits(1))

//CONSTRUCTION ET APPLICATION DU MODELE AU JEU D'APPRENTISSAGE
val model = new LogisticRegressionWithLBFGS()
  .setNumClasses(2)
  .run(trainingData)

// Sauvegarde du modèle
model.save(sc, "modele/LGB_W2V")

//APPLICATION DU MODELE AU JEU DE VALIDATION
val predictionAndLabels = validationData.map { case LabeledPoint(label, features) =>
  val prediction = model.predict(features)
  (prediction, label)
}

//CALCUL DES INDICATEURS DE PERFORMANCE
val metrics_LGB_W2V = new BinaryClassificationMetrics(predictionAndLabels)
val auPRC_LGB_W2V = 100 * (metrics_LGB_W2V.areaUnderPR() - metrics_LGB_W2V.areaUnderPR() % 0.0001) 
val auROC_LGB_W2V = 100 * (metrics_LGB_W2V.areaUnderROC() - metrics_LGB_W2V.areaUnderROC() % 0.0001) 

//LECTURE D'UN FICHIER AU FORMAT LIBSVM
val echantillon_LIBSVM: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "Data/Vecto_Word2vecC2")

//PARTAGE DE L'ECHANTILLON EN JEUX D'APPRENTISSAGE ET DE VALIDATION
val splits = echantillon_LIBSVM.randomSplit(Array(0.7, 0.3))
val (trainingData, validationData) = (splits(0), splits(1))

//CONSTRUCTION ET APPLICATION DU MODELE AU JEU D'APPRENTISSAGE
val model = new LogisticRegressionWithLBFGS()
  .setNumClasses(2)
  .run(trainingData)

// Sauvegarde du modèle
model.save(sc, "modele/LGB_W2VC2")

//APPLICATION DU MODELE AU JEU DE VALIDATION
val predictionAndLabels = validationData.map { case LabeledPoint(label, features) =>
  val prediction = model.predict(features)
  (prediction, label)
}


//CALCUL DES INDICATEURS DE PERFORMANCE
val metrics_LGB_W2V_C2 = new BinaryClassificationMetrics(predictionAndLabels)
val auPRC_LGB_W2V_C2 = 100 * (metrics_LGB_W2V_C2.areaUnderPR() - metrics_LGB_W2V_C2.areaUnderPR() % 0.0001) 
val auROC_LGB_W2V_C2 = 100 * (metrics_LGB_W2V_C2.areaUnderROC() - metrics_LGB_W2V_C2.areaUnderROC() % 0.0001) 

//LECTURE D'UN FICHIER AU FORMAT LIBSVM
val echantillon_LIBSVM: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "Data/Vecto_Countvectorizer")

//PARTAGE DE L'ECHANTILLON EN JEUX D'APPRENTISSAGE ET DE VALIDATION
val splits = echantillon_LIBSVM.randomSplit(Array(0.7, 0.3))
val (trainingData, validationData) = (splits(0), splits(1))

//CONSTRUCTION ET APPLICATION DU MODELE AU JEU D'APPRENTISSAGE
val model = new LogisticRegressionWithLBFGS()
  .setNumClasses(2)
  .run(trainingData)
// Sauvegarde du modèle
model.save(sc, "modele/LGB_CV")

//APPLICATION DU MODELE AU JEU DE VALIDATION
val predictionAndLabels = validationData.map { case LabeledPoint(label, features) =>
  val prediction = model.predict(features)
  (prediction, label)
}

//CALCUL DES INDICATEURS DE PERFORMANCE
val metrics_LGB_CV = new BinaryClassificationMetrics(predictionAndLabels)
val auPRC_LGB_CV = 100 * (metrics_LGB_CV.areaUnderPR() - metrics_LGB_CV.areaUnderPR() % 0.0001) 
val auROC_LGB_CV = 100 * (metrics_LGB_CV.areaUnderROC() - metrics_LGB_CV.areaUnderROC() % 0.0001) 

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics



//LECTURE D'UN FICHIER AU FORMAT LIBSVM
val echantillon_LIBSVM :RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "Data/Vecto_HTF")

// Split data into training (70%) and test (30%).
val Array(trainingData, validationData) = echantillon_LIBSVM.randomSplit(Array(0.7, 0.3))

val model = NaiveBayes.train(trainingData, lambda = 1.0, modelType = "multinomial")

// Sauvegarde du modèle
model.save(sc, "modele/ANB_HTF")

//APPLICATION DU MODELE AU JEU DE VALIDATION
val predictionAndLabels = validationData.map { case LabeledPoint(label, features) =>
  val prediction = model.predict(features)
  (prediction, label)
}
//val predictionAndLabel = test.map(p => (modele.predict(p.features), p.label))

//CALCUL DES INDICATEURS DE PERFORMANCE
val metrics_ANB_HTF = new BinaryClassificationMetrics(predictionAndLabels)
val auPRC_ANB_HTF = 100 * (metrics_ANB_HTF.areaUnderPR() - metrics_ANB_HTF.areaUnderPR() % 0.0001) 
val auROC_ANB_HTF = 100 * (metrics_ANB_HTF.areaUnderROC() - metrics_ANB_HTF.areaUnderROC() % 0.0001) 

//LECTURE D'UN FICHIER AU FORMAT LIBSVM
val echantillon_LIBSVM :RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "Data/Vecto_Countvectorizer")

// Split data into training (70%) and test (30%).
val Array(trainingData, validationData) = echantillon_LIBSVM.randomSplit(Array(0.7, 0.3))

val model = NaiveBayes.train(trainingData, lambda = 1.0, modelType = "multinomial")

// Sauvegarde du modèle
model.save(sc, "modele/ANB_CV")

//APPLICATION DU MODELE AU JEU DE VALIDATION
val predictionAndLabels = validationData.map { case LabeledPoint(label, features) =>
  val prediction = model.predict(features)
  (prediction, label)
}

//CALCUL DES INDICATEURS DE PERFORMANCE
val metrics_ANB_CV = new BinaryClassificationMetrics(predictionAndLabels)
val auPRC_ANB_CV = 100 * (metrics_ANB_CV.areaUnderPR() - metrics_ANB_CV.areaUnderPR() % 0.0001) 
val auROC_ANB_CV = 100 * (metrics_ANB_CV.areaUnderROC() - metrics_ANB_CV.areaUnderROC() % 0.0001) 

//Affichage des métriques
println("Modèle GRADIENT BOOSTING - Vectorisation HASHINGTF")
println("============================================================")
println(s"Area under precision-recall curve = $auPRC_GBT_HTF %")
println(s"Area under ROC = $auROC_GBT_HTF %")
println("Modèle GRADIENT BOOSTING - Vectorisation WORD2VEC sur corpus wikipédia")
println("============================================================")
println(s"Area under precision-recall curve = $auPRC_GBT_W2V %")
println(s"Area under ROC = $auROC_GBT_W2V %")
println("Modèle GRADIENT BOOSTING - Vectorisation WORD2VEC sur corpus commentaire")
println("============================================================")
println(s"Area under precision-recall curve = $auPRC_GBT_W2V_C2 %")
println(s"Area under ROC = $auROC_GBT_W2V_C2 %")
println("Modèle GRADIENT BOOSTING - Vectorisation COUNTVECTORIZER")
println("============================================================")
println(s"Area under precision-recall curve = $auPRC_GBT_CV %")
println(s"Area under ROC = $auROC_GBT_CV %")
println("Modèle SVM - vectorisation HASHINGTF")
println("============================================================")
println(s"Area under precision-recall curve = $auPRC_SVM_HTF %")
println(s"Area under ROC = $auROC_SVM_HTF %")
println("Modèle SVM - vectorisation WORD2VEC sur corpus wikipédia")
println("============================================================")
println(s"Area under precision-recall curve = $auPRC_SVM_W2V %")
println(s"Area under ROC = $auROC_SVM_W2V %")
println("Modèle SVM - vectorisation WORD2VEC sur corpus commentaire")
println("============================================================")
println(s"Area under precision-recall curve = $auPRC_SVM_W2V_C2 %")
println(s"Area under ROC = $auROC_SVM_W2V_C2 %")
println("Modèle SVM - vectorisation COUNTVECTORIZER")
println("============================================================")
println(s"Area under precision-recall curve = $auPRC_SVM_CV %")
println(s"Area under ROC = $auROC_SVM_CV %")
println("Modèle REGRESSION LOGISTIQUE - vectorisation HASHINGTF")
println("============================================================")
println(s"Area under precision-recall curve = $auPRC_LGB_HTF %")
println(s"Area under ROC = $auROC_LGB_HTF %")
println("Modèle REGRESSION LOGISTIQUE - vectorisation WORD2VEC sur corpus wikipédia")
println("============================================================")
println(s"Area under precision-recall curve = $auPRC_LGB_W2V %")
println(s"Area under ROC = $auROC_LGB_W2V %")
println("Modèle REGRESSION LOGISTIQUE - vectorisation WORD2VEC sur corpus commentaire")
println("============================================================")
println(s"Area under precision-recall curve = $auPRC_LGB_W2V_C2 %")
println(s"Area under ROC = $auROC_LGB_W2V_C2 %")
println("Modèle REGRESSION LOGISTIQUE - vectorisation COUNTVECTORIZER")
println("============================================================")
println(s"Area under precision-recall curve = $auPRC_LGB_CV %")
println(s"Area under ROC = $auROC_LGB_CV %")
println("Modèle NAIVE BAYES - vectorisation HASHINGTF")
println("============================================================")
println(s"Area under precision-recall curve = $auPRC_ANB_HTF %")
println(s"Area under ROC = $auROC_ANB_HTF %")
println("Modèle NAIVE BAYES - vectorisation COUNTVECTORIZER")
println("============================================================")
println(s"Area under precision-recall curve = $auPRC_ANB_CV %")
println(s"Area under ROC = $auROC_ANB_CV %")

