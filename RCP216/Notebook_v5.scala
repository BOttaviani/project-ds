//Aller dans le répertoire cible où se trouvent les données à traiter

cd Documents/Big_Data/RCP216/Projet/Données_projet



//Lancement de spark-shell avec graphframes (après avoir, au préalable, téléchargé le package graphframe sur https://spark-packages.org/package/graphframes/graphframes)

spark-shell --packages graphframes:graphframes:0.5.0-spark2.1-s_2.11



//Le code SCALA commence ici ...




-------------------------------------------------------------------------------------------------------------------------
//IMPORTER LA BIBLIOTHEQUE SQL SPARK
import org.apache.spark.sql.types._

//IMPORTER LA BIBLIOTHEQUE GRAPHFRAME
import org.graphframes._

//IMPORTER LA BIBLIOTHEQUE GRAPHX
import org.apache.spark.graphx._

//DEFINITION DE LA STRUCTURE DU FICHIER DES CODES IATA
val codes_IATA_struct = StructType(StructField("Code_IATA", StringType, true) :: StructField("nom", StringType, true) :: StructField("code_pays", StringType, true) :: StructField("ville", StringType, true) :: Nil)

//DEFINITION DE LA STRUCTURE DU FICHIER D'ENTREE POUR LES NOEUDS
val aeroports_struct_noeuds = StructType(StructField("Code_comp", StringType, true) :: StructField("num_comp", StringType, true) :: StructField("code_source", StringType, true) :: StructField("id", StringType, true) :: StructField("code_dest", StringType, true) :: StructField("num_dest", StringType, true) :: StructField("filler", StringType, true) :: StructField("nb_etapes", ShortType, true) :: StructField("code", StringType, true) :: Nil)

//DEFINITION DE LA STRUCTURE DU FICHIER D'ENTREE POUR LES LIENS
val aeroports_struct_liens = StructType(StructField("Code_comp", StringType, true) :: StructField("num_comp", StringType, true) :: StructField("code_source", StringType, true) :: StructField("src", StringType, true) :: StructField("code_dest", StringType, true) :: StructField("dst", StringType, true) :: StructField("filler", StringType, true) :: StructField("nb_etapes", ShortType, true) :: StructField("code", StringType, true) :: Nil)

//LECTURE DU FICHIER D'ENTREE DES CODES IATA AVEC LA STRUCTURE ASSOCIEE DEFINIE PLUS HAUT
val codes_IATA = spark.read.format("csv").schema(codes_IATA_struct).load("Codes_IATA")

//LECTURE DU FICHIER D'ENTREE A L'AIDE DE LA STRUCTURE DES NOEUDS DEFINIE PLUS HAUT
val aeroports_noeuds = spark.read.format("csv").schema(aeroports_struct_noeuds).load("routes.dat")

//CREATION DE LA TABLE ASSOCIEE AUX CODES IATA
codes_IATA.createOrReplaceTempView("codes_IATA")

//CREATION DE LA TABLE ASSOCIEE AUX NOEUDS
aeroports_noeuds.createOrReplaceTempView("aeroports_noeuds")

//SELECTION DES DONNEES SERVANT A LA CONSTRUCTION DU GRAPHE A PARTIR DE LA TABLE DES NOEUDS
val Noeuds = spark.sql("select distinct id, code_source from (select id, code_source from aeroports_noeuds where id not like '%N' union select num_dest, code_dest from aeroports_noeuds where num_dest not like '%N')")
Noeuds.createOrReplaceTempView("Noeuds")

//DETERMINATION DE L'ORDRE DU GRAPHE - COMPTAGE DES NOEUDS
spark.sql("select count(*) as nb_Noeuds from Noeuds").show

//LECTURE DU FICHIER D'ENTREE A L'AIDE DE LA STRUCTURE DES LIENS DEFINIE PLUS HAUT
val aeroports_liens = spark.read.format("csv").schema(aeroports_struct_liens).load("routes.dat")

//CREATION DE LA TABLE ASSOCIEE AUX LIENS
aeroports_liens.createOrReplaceTempView("aeroports_liens")

val Liens = spark.sql("select distinct a.src, a.dst from aeroports_liens a where src not like '%N' and dst not like '%N'")
Liens.createOrReplaceTempView("Liens")

//COMPTAGE DES LIENS
spark.sql("select count(*) as nb_Liens from Liens").show

//CONSTRUCTION DU GRAPHE
val Graphe_aeroports = GraphFrame(Noeuds, Liens)

//POIDS DES LIENS
//1a- Sans le nom des villes 
val Poids_Liens = spark.sql("select a.depart, b.ville, a.arrivee, c.ville, a.poids from(select code_source as depart, code_dest as arrivee, count(*) as poids from aeroports_liens where src not like '%N' and dst not like '%N' group by depart, arrivee order by poids desc, depart asc) as a")
Poids_Liens_orientes.createOrReplaceTempView("Poids_Liens_orientes")

//1b- Avec le nom des villes 
val Poids_Liens_orientes = spark.sql("select a.depart, b.ville, a.arrivee, c.ville, a.poids from(select code_source as depart, code_dest as arrivee, count(*) as poids from aeroports_liens where src not like '%N' and dst not like '%N' group by depart, arrivee order by poids desc, depart asc) as a inner join codes_IATA b on b.code_IATA = a.depart inner join codes_IATA c on c.code_IATA = a.arrivee")
Poids_Liens_orientes.createOrReplaceTempView("Poids_Liens_orientes")
//Statistiques élémentaires
spark.sql("select poids from Poids_Liens").describe().show
//Top 5
Poids_Liens_orientes.show(5)

//2- CALCUL DE LA DENSITE
val nb_noeuds = spark.sql("select cast(count(*) as decimal(7,2)) as nb_noeuds from Noeuds")
val nb_liens = spark.sql("select cast(count(*) as decimal(7,2)) as nb_liens from Liens")
val table_densite = nb_noeuds.crossJoin(nb_liens)
table_densite.createOrReplaceTempView("table_densite")
//Résultat du calcul
spark.sql("select cast((nb_liens)/(nb_noeuds*(nb_noeuds-1)) as decimal (5,4)) as densite from table_densite").show

//3- DISTRIBUTIONS DES DEGRES (PLUS PROCHES VOISINS)
val degres_entrants = Graphe_aeroports.inDegrees
degres_entrants.createOrReplaceTempView("degres_entrants")
val degres_sortants = Graphe_aeroports.outDegrees
degres_sortants.createOrReplaceTempView("degres_sortants")
val nb_voisins = Graphe_aeroports.degrees
nb_voisins.createOrReplaceTempView("nb_voisins")
//Degres entrants et sortants, car graphe orienté, et plus proches voisins
//Statistiques élémentaires
spark.sql("select inDegree as degres_entrants from degres_entrants").describe().show
spark.sql("select outDegree as degres_sortants from degres_sortants").describe().show
spark.sql("select degree as plus_proches_voisins from nb_voisins").describe().show
//Seuls 3 aéroports n'ont pas de plus proches voisins, car ils sont en lien
//avec des aéroports non identifiés
spark.sql("select a.id, a.code_source as IATA_sans_voisin from Noeuds a left outer join nb_voisins b on b.id = a.id where b.id is null").show
//Tops 5 des degrés entrants, sortants, et du nombre de voisins
spark.sql("select a.code_source as code_IATA, b.inDegree as degres_entrants from Noeuds a inner join degres_entrants b on b.id = a.id order by b.inDegree desc").show(5)
spark.sql("select a.code_source as code_IATA, b.outDegree as degres_sortants from Noeuds a inner join degres_sortants b on b.id = a.id order by b.outDegree desc").show(5)
spark.sql("select a.code_IATA, b.ville, a.nb_voisins from(select a.code_source as code_IATA, b.degree as nb_voisins from Noeuds a inner join nb_voisins b on b.id = a.id order by b.degree desc) as a inner join codes_IATA as b on b.code_IATA = a.code_IATA").show(5)

//4- COMPOSANTES CONNEXES DANS LE RESEAU
val GrapheX_aeroports = Graphe_aeroports.toGraphX
val composantes_connexes = GrapheX_aeroports.connectedComponents
def sortedConnectedComponents(connectedComponents: Graph[VertexId, _]): Seq[(VertexId, Long)] = {
  val comptage_composantes = connectedComponents.vertices.map(_._2).countByValue
  comptage_composantes.toSeq.sortBy(_._2).reverse
}
val comptage_composantes = sortedConnectedComponents(composantes_connexes)
comptage_composantes.size
comptage_composantes.take(comptage_composantes.size).foreach(println)
val composante_connexe_principale = comptage_composantes.take(1).toList.toDF
composante_connexe_principale.createOrReplaceTempView("composante_connexe_principale")
val nb_noeuds_connectes = spark.sql("select _2 as nb_noeuds_connectes from composante_connexe_principale")
nb_noeuds_connectes.show
val pourcentage_noeuds_composante_principale = nb_noeuds_connectes.crossJoin(nb_noeuds)
pourcentage_noeuds_composante_principale.createOrReplaceTempView("pourcentage_noeuds_composante_principale")
spark.sql("select cast((nb_noeuds_connectes/nb_noeuds)*100 as decimal(4,2)) as pourcentage_noeuds_connectes from pourcentage_noeuds_composante_principale").show


//5- CENTRALITES - RECHERCHE DE PLUS COURTS TRAJETS AVEC AEROPORTS INTERMEDIAIRES
//Recherche de tous les trajets impliquant 3 aéroports
val trajets = Graphe_aeroports.find("(a)-[]->(b); (b)-[]->(c)")
trajets.createOrReplaceTempView("trajets")
//Récupération des aéroports intermédiaires sur le plus court trajet (orientation intégrée dans liens : on cumule tous les chemins dans tous les sens)
val intermediaires = spark.sql("select b.code_source from trajets where a<>b and b<>c")
intermediaires.createOrReplaceTempView("intermediaires")
//Calcul des centralites (nécessaire pour le calcul du clustering plus bas)
val centralites = spark.sql("select a.code_IATA, b.ville, a.centralite from(select code_source as code_IATA, count(*) as centralite from intermediaires group by code_source order by centralite desc) as a inner join codes_IATA b on b.code_IATA = a.code_IATA")
centralites.createOrReplaceTempView("centralites")
//Statistiques élémentaires
spark.sql("select centralite from centralites").describe().show
//Top 5
centralites.show(5)

//6- CLUSTERING (pour rester cohérents avec la centralité, nous prenons en compte toutes les orientations)
//Détermination/distribution du nombre de triangles
val triangles = Graphe_aeroports.triangleCount.run
triangles.createOrReplaceTempView("triangles")
//Statistiques élémentaires
spark.sql("select count as nb_triangles from triangles").describe().show
//Top 5
spark.sql("select a.code_IATA, b.ville, a.nb_triangles from (select id, code_source as code_IATA, count as nb_triangles from triangles order by count desc) as a inner join codes_IATA b on b.code_IATA = a.code_IATA").show(5)
//Clustering global
val somme_triangles = spark.sql("select sum(count) as total_triangles from triangles")
val somme_triades_simples = spark.sql("select sum(centralite) as total_triades from centralites")
val clustering_global = somme_triangles.crossJoin(somme_triades_simples)
clustering_global.createOrReplaceTempView("clustering_global")
spark.sql("select cast((3*total_triangles)/(total_triades+(3*total_triangles)) as decimal(5,4)) as clustering_global from clustering_global").show
//Clustering local pour chaque aéroport
val nb_triangles = spark.sql("select id, count as nb_triangles from triangles")
nb_triangles.createOrReplaceTempView("nb_triangles")
val clustering_local = spark.sql("select a.id, a.nb_triangles, b.degree, cast((2*a.nb_triangles)/(degree*(degree-1)) as decimal (5,4)) as clustering_local from nb_triangles a inner join nb_voisins b on b.id = a.id")
clustering_local.createOrReplaceTempView("clustering_local")
//Statistiques élémentaires
spark.sql("select clustering_local from clustering_local").describe().show
//Top 5 en fonction du nombre de triangles et du nombre de voisins (on prend le 1/3 supérieur pour chacune des variables)
spark.sql("select a.code_IATA, b.ville, a.nb_voisins, a.nb_triangles, a.clustering_local from (select a.id, a.code_source as code_IATA, b.degree as nb_voisins, b.nb_triangles as nb_triangles, b.clustering_local from Noeuds a inner join clustering_local b on b.id = a.id where b.degree > 318 and b.nb_triangles > 3029 order by clustering_local desc) as a inner join codes_IATA b on b.code_IATA = a.code_IATA").show(5)

//7- LPA (LABEL PROPAGATION) - DETERMINATION DE L'EXISTENCE DE COMMUNAUTES
val LPA = Graphe_aeroports.labelPropagation.maxIter(5).run()
LPA.createOrReplaceTempView("LPA")
//Table des communautés
val communautes = spark.sql("select label, count(*) as nbre from LPA group by label order by nbre desc")
communautes.createOrReplaceTempView("communautes")
//Statistiques élémentaires
spark.sql("select nbre from communautes").describe().show
//Top 5
communautes.show(5)
//549755813898 (736)  => Europe
spark.sql("select distinct l.id, a.code_source as code_IATA, d.ville, d.code_pays, c.degree as nb_voisins from LPA l inner join aeroports_noeuds a on a.id = l.id inner join nb_voisins c on c.id = l.id inner join codes_IATA d on d.code_IATA = a.code_source where label = 549755813898 order by nb_voisins desc").show
//1030792151047 (675) => Amérique du Nord
spark.sql("select distinct l.id, a.code_source as code_IATA, d.ville, d.code_pays, c.degree as nb_voisins from LPA l inner join aeroports_noeuds a on a.id = l.id inner join nb_voisins c on c.id = l.id inner join codes_IATA d on d.code_IATA = a.code_source where label = 1030792151047 order by nb_voisins desc").show
//790273982469 (211)  => Chine
spark.sql("select distinct l.id, a.code_source as code_IATA, d.ville, d.code_pays, c.degree as nb_voisins from LPA l inner join aeroports_noeuds a on a.id = l.id inner join nb_voisins c on c.id = l.id inner join codes_IATA d on d.code_IATA = a.code_source where label = 790273982469 order by nb_voisins desc").show
//206158430212 (123)  => Moyen Orient - Inde - Pakistan
spark.sql("select distinct l.id, a.code_source as code_IATA, d.ville, d.code_pays, c.degree as nb_voisins from LPA l inner join aeroports_noeuds a on a.id = l.id inner join nb_voisins c on c.id = l.id inner join codes_IATA d on d.code_IATA = a.code_source where label = 206158430212 order by nb_voisins desc").show
//51539607564 (110)   => Asie du Nord (Russie et anciennes républiques de l'Union soviétique) 
spark.sql("select distinct l.id, a.code_source as code_IATA, d.ville, d.code_pays, c.degree as nb_voisins from LPA l inner join aeroports_noeuds a on a.id = l.id inner join nb_voisins c on c.id = l.id inner join codes_IATA d on d.code_IATA = a.code_source where label = 51539607564 order by nb_voisins desc").show
//Le nombre total de labels est égal au nombre de noeuds du réseau
spark.sql("select sum(nbre) from(select label, count(*) as nbre from LPA group by label order by nbre desc)").show

//8- TABLES POUR CALCUL MODULARITE
//Construction de la table rassemblant les labels de communauté, les noeuds, et les degrés
val noeuds_enrichis = spark.sql("select a.label, b.id as aeroport, c.degree as degre from LPA a inner join Noeuds b on b.id = a.id inner join nb_voisins c on c.id = a.id")
noeuds_enrichis.createOrReplaceTempView("noeuds_enrichis")
//Calcul de la double somme sur égalité des labels de communauté pour des noeuds différents
val double_somme = spark.sql("select a.label as label_depart, a.aeroport as depart, a.degre as degre_depart, b.label as label_arrivee, b.aeroport as arrivee, b.degre as degre_arrivee from noeuds_enrichis a inner join noeuds_enrichis b on b.label = a.label and a.aeroport <> b.aeroport") 
double_somme.createOrReplaceTempView("double_somme")
//Ajout du nombre de liens à la double somme pour calcul du produit
val double_somme_enrichie = double_somme.crossJoin(nb_liens)
double_somme_enrichie.createOrReplaceTempView("double_somme_enrichie")
//Construction de la table des produits
val produits = spark.sql("select a.*, cast((degre_depart * degre_arrivee )/(2*nb_liens) as decimal(7,2)) as produit from double_somme_enrichie a") 
produits.createOrReplaceTempView("produits")
//Construction de la somme des produits
val somme_produits = spark.sql("select sum(produit) as somme_produits from produits")
somme_produits.createOrReplaceTempView("somme_produits")
val delta  = somme_produits.crossJoin(nb_liens)
delta.createOrReplaceTempView("delta")
//Le nombre Aij est égal au double du nombre de liens, étant donné que les couples
//sont comptés 2 fois dans la double somme
spark.sql("select cast((2*nb_liens - somme_produits)/(2*nb_liens) as decimal(5,4)) as modularite from delta").show

