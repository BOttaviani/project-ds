create class Produit extends V;
create class Client extends V;
create class Panier extends V;
create class Commentaire extends V;
create class HasProduit extends E;
create class HasComment extends E;

create property Produit.p_id Integer;
create property Produit.nom String;
create property Produit.categorie String;
create property Produit.prix Float;
create property Produit.list_comment LINKLIST Commentaire;
create index Produit.p_id UNIQUE;

insert into Produit(p_id,nom,categorie,prix) values (1,"TV Samsung LCD22","TV",599.99);
insert into Produit(p_id,nom,categorie,prix) values (2,"TV Panasonic HD21","TV",399.90);
insert into Produit(p_id,nom,categorie,prix) values (3,"Apple X","Smartphone",999.99);
insert into Produit(p_id,nom,categorie,prix) values (4,"Samsung 8","Smartphone",790);
insert into Produit(p_id,nom,categorie,prix) values (5,"Lenovo T430S","Ordinateur",599.99);

create property Client.c_id Integer;
create property Client.nom String;
create property Client.prenom String;
create index  Client.c_id UNIQUE;

insert into Client (c_id,nom,prenom) values (1,"OTTAVIANI","Bernard");
insert into Client (c_id,nom,prenom) values (2,"DUPONT","Denis");
insert into Client (c_id,nom,prenom) values (3,"RIGAUX","Philippe");

create property Panier.a_id Integer;
create property Panier.date_creation Date;
create property Panier.lien_client LINK Client;

create property HasProduit.quantite Integer;

insert into Panier set a_id=1,date_creation='2018-01-19',lien_client=(select from Client where c_id=1);
create edge HasProduit from (select from Produit where p_id=2) to (select from Panier where a_id=1) set quantite = 1;
create edge HasProduit from (select from Produit where p_id=4) to (select from Panier where a_id=1) set quantite = 2;
create edge HasProduit from (select from Produit where p_id=5) to (select from Panier where a_id=1) set quantite = 1;

insert into Panier set a_id=3,date_creation='2018-01-22',lien_client=(select from Client where c_id=2);
create edge HasProduit from (select from Produit where p_id=1) to (select from Panier where a_id=3) set quantite = 1;
create edge HasProduit from (select from Produit where p_id=3) to (select from Panier where a_id=3) set quantite = 1;

insert into Panier set a_id=2,date_creation='2018-01-21',lien_client=(select from Client where c_id=3);
create edge HasProduit from (select from Produit where p_id=5) to (select from Panier where a_id=2) set quantite = 3;

create property Commentaire.cm_id Integer;
create property Commentaire.date_cm Date;
create property Commentaire.texte_cm String;
create property Commentaire.note Integer;
create property Commentaire.p_id Integer;
create property Commentaire.lien_client LINK Client;

insert into Commentaire set cm_id=1, date_cm='2018-01-02',texte_cm="Pas mal",note=8,p_id=4,lien_client=(select from Client where c_id=1);
create edge HasComment from (select from Produit where p_id=4) to (select from Commentaire where cm_id=1);
insert into Commentaire set cm_id=2, date_cm='2018-01-05',texte_cm="Bof",note=4,p_id=3,lien_client=(select from Client where c_id=2);
create edge HasComment from (select from Produit where p_id=3) to (select from Commentaire where cm_id=2);
insert into Commentaire set cm_id=3, date_cm='2018-01-12',texte_cm="Top",note=9,p_id=5,lien_client=(select from Client where c_id=1);
create edge HasComment from (select from Produit where p_id=5) to (select from Commentaire where cm_id=3);
insert into Commentaire set cm_id=4, date_cm='2018-01-22',texte_cm="Formidable",note=10,p_id=5,lien_client=(select from Client where c_id=3);
create edge HasComment from (select from Produit where p_id=5) to (select from Commentaire where cm_id=4);

update Produit set list_comment = (select @rid from Commentaire where p_id = 1) where p_id=1;
update Produit set list_comment = (select @rid from Commentaire where p_id = 2) where p_id=2;
update Produit set list_comment = (select @rid from Commentaire where p_id = 3) where p_id=3;
update Produit set list_comment = (select @rid from Commentaire where p_id = 4) where p_id=4;
update Produit set list_comment = (select @rid from Commentaire where p_id = 5) where p_id=5;

