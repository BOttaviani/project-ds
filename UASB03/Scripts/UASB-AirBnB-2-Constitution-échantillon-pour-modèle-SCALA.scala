
//LECTURE DU FICHIER D'ENTREE DES COMMENTAIRES AVEC DELIMITEUR £
val commentaires_lus = spark.read.option("delimiter", "£").option("charset", "utf-8").csv("Data/Commentaires_identifiants_V3")

//REFORMATAGE DES COMMENTAIRES LUS => TOUS LES CARACTERES SONT MIS EN LETTRES MINUSCULES
val commentaires = commentaires_lus.rdd.map(x =>(x.getAs[String](0), x.getAs[String](1).toLowerCase)).toDF("Id_comment", "commentaire")

//CREATION DE LA TABLE ASSOCIEE AUX COMMENTAIRES BRUTS
commentaires.createOrReplaceTempView("commentaires_bruts")


//NOMBRE DE COMMENTAIRES
var commentaires_nbre = commentaires.count()

//LECTURE DU FICHIER D'EVALUATION
val evaluation_brute = spark.read.option("delimiter", "#").csv("Data/Evaluation_V4")

//CREATION DE LA TABLE ASSOCIEE A L'EVALUATION LUE
evaluation_brute.createOrReplaceTempView("evaluation_brute")

//CREATION DU NOM DES COLONNES POUR LA TABLE EVALUATION
val evaluation = spark.sql("select _c0 as expression, _c1 as poids from evaluation_brute")

//CREATION DE LA TABLE ASSOCIEE A L'EVALUATION FINALISEE
evaluation.createOrReplaceTempView("evaluation")

//PRODUIT CARTESIEN DES TABLES COMMENTAIRE ET EVALUATION
val produit_brut = commentaires.crossJoin(evaluation)

//CREATION DE LA TABLE ASSOCIEE AU PRODUIT CARTESIEN BRUT 
produit_brut.createOrReplaceTempView("produit_brut")

//PRODUIT CARTESIEN RAFFINE
val produit = spark.sql("select Id_comment, commentaire, poids from produit_brut where commentaire like expression")

//CREATION DE LA TABLE ASSOCIEE AU PRODUIT CARTESIEN RAFFINE 
produit.createOrReplaceTempView("produit")

//CALCUL DE L'EVALUATION POUR CHAQUE COMMENTAIRE
val commentaires_evalues_bruts = spark.sql("select Id_comment, commentaire, sum(poids) as evaluation from produit group by Id_comment, commentaire")

//CREATION DE LA TABLE COMMENTAIRES EVALUES BRUT
commentaires_evalues_bruts.createOrReplaceTempView("commentaires_evalues_bruts")

//EVALUATION DEFINITIVE DES COMMENTAIRES
val commentaires_evalues = spark.sql("select Id_comment, commentaire, case when evaluation > 0 then 'positif' else case when evaluation < 0 then 'negatif' else 'neutre' end end as qualite from commentaires_evalues_bruts")

//CREATION DE LA TABLE COMMENTAIRES EVALUES
commentaires_evalues.createOrReplaceTempView("commentaires_evalues")


//COMMENTAIRES POSITIFS
//Lecture
//var commentaires_positifs = spark.sql("select Id_comment, commentaire, qualite from commentaires_evalues where qualite = 'positif' and length(commentaire) <= "+Q3S)
var commentaires_positifs = spark.sql("select Id_comment, commentaire, qualite from commentaires_evalues where qualite = 'positif'")
//Comptage
val commentaires_positifs_nbre = commentaires_positifs.count()

//COMMENTAIRES NEGATIFS
//Lecture
//var commentaires_negatifs = spark.sql("select Id_comment, commentaire, qualite from commentaires_evalues where qualite = 'negatif' and length(commentaire) <= "+Q3S)
var commentaires_negatifs = spark.sql("select Id_comment, commentaire, qualite from commentaires_evalues where qualite = 'negatif'")
//Comptage
val commentaires_negatifs_nbre = commentaires_negatifs.count()

//DEFINITION DE LA TAILLE DE L'ECHANTILLON D'APPRENTISSAGE
val taille_echantillon = Math.min(commentaires_negatifs_nbre, commentaires_positifs_nbre).toInt
val taille_echantillon_positif = 5 * taille_echantillon
//DEFINITION DE L'ECHANTILLON D'APPRENTISSAGE AVEC AUTANT DE COMMENTAIRES POSITIFS QUE NEGATIFS
var echantillon_evalue = commentaires_positifs.limit(taille_echantillon_positif).unionAll(commentaires_negatifs.limit(taille_echantillon))
echantillon_evalue.createOrReplaceTempView("echantillon_evalue")

echantillon_evalue.coalesce(1).write.format("com.databricks.spark.csv").option("header","true").save("Data/echantillon_evalue")


println(s"Nombre de commentaires positifs :$commentaires_positifs_nbre")
println(s"Nombre de commentaires negatifs :$commentaires_negatifs_nbre")
println(s"Nombre de commentaires positifs  dans l'échantillon :$taille_echantillon_positif")
println(s"Nombre de commentaires negatifs  dans l'échantillon :$taille_echantillon")

