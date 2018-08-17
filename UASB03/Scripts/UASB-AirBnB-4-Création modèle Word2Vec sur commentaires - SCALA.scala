
import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}

val input = sc.textFile("Data/frwiki_formate3.txt").map(line => line.split(" ").toSeq)
val word2vec = new Word2Vec()
val model = word2vec.fit(input)

// Sauvegarde du modèle
model.save(sc, "modele/Word2VecFR_complet")


val input = sc.textFile("Data/corpus_commentaire.csv").map(line => line.split(" ").toSeq)
val word2vec = new Word2Vec()
val model = word2vec.fit(input)

// Sauvegarde du modèle
model.save(sc, "modele/Word2VecFR_comment")


val synonyms = model.findSynonyms("quartier", 5)

for((synonym, cosineSimilarity) <- synonyms) {
  println(s"$synonym $cosineSimilarity")
}

