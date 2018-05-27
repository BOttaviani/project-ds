
#Définition des librairies
library(jsonlite)
library(Hmisc)
library(FactoMineR)
library(rpart)
library(rpart.plot)

#Chargement du jeu de données
path="/home/user/Documents/STA211-Datamining/Projet_STA/"
Airbnb<-fromJSON(paste(path,"airbnb_V3.json",sep=""))
sapply(Airbnb,class)
summary(Airbnb)


character_vars <- lapply(Airbnb, class) == "character"
Airbnb[, character_vars] <- lapply(Airbnb[, character_vars], as.factor)
#définition de la variable arrondissement comme qualitative
Airbnb$arrondissement <- as.factor(Airbnb$arrondissement ) 
sapply(Airbnb,class)

#suppression de la variable Prix total
Airbnb <- subset(Airbnb, select = -c(total_price) )

# Arbre sur la variable price_level
AirbnbTree <- rpart(price_level~.,data=Airbnb,control=rpart.control(minsplit=50,cp = 0.002, xval=100))
#AirbnbTree <- rpart(price_level~.,data=Airbnb,control=rpart.control(minsplit=500,cp = 0.01, xval=20))
printcp(AirbnbTree)


bestcp <- AirbnbTree$cptable[which.min(AirbnbTree$cptable[,"xerror"]),"CP"]

# Step3: Prune the tree using the best cp.
Airbnb_pruned <- prune(AirbnbTree, cp = bestcp)
printcp(Airbnb_pruned)

plotcp(AirbnbTree)

prp(Airbnb_pruned,extra=1)

prp(AirbnbTree, main="Explication du niveau de prix",
    extra=103,           # display prob of survival and percent of obs
    nn=TRUE,             # display the node numbers
    fallen.leaves=TRUE,  # put the leaves on the bottom of the page
    shadow.col="gray",   # shadows under the leaves
    branch.lty=3,        # draw branches using dotted lines
    branch=.5,           # change angle of branch lines
    yesno=2,
    xflip=TRUE,
    faclen=0,            # faclen=0 to print full factor names
    trace=1,             # print the automatically calculated cex
    split.cex=1.2,       # make the split text larger than the node text
#    split.prefix="is ",  # put "is " before split text
    split.suffix=" ?",    # put "?" after split text
#    col=cols, border.col=cols,   # green if survived
    split.box.col="lightgray",   # lightgray split boxes (default is white)
    split.border.col="darkgray", # darkgray border on split boxes
    split.round=.5)              # round the split box corners a tad


library(partykit)
plot(as.party(AirbnbTree),  tp_args = list(id = FALSE))

jpeg("/home/user/Documents/STA211-Datamining/Projet_STA/Tree1_level_2.jpg", width=2000, height=750)
plot.new()
plot(as.party(AirbnbTree),   tp_args = list(FUN = function(info)
  format(round(info$prediction, digits = 1), nsmall = 1)
))
title(main="Arbre de décision sur Price_level", cex.main=3, line=1) 
dev.off()


library(RColorBrewer)
library(rattle)
jpeg("/home/user/Documents/STA211-Datamining/Projet_STA/Tree2_level_2.jpg", width=2000, height=750)
plot.new()
fancyRpartPlot(AirbnbTree, sub = '')
title(main="Arbre de décision sur Price_level", cex.main=3, line=1)

dev.off()

pred=predict(object = AirbnbTree,newdata = Airbnb,type="class")
mc<-table(Airbnb$price_level,pred)
print(mc)
rowSums(mc)
rowSums(mc)[2]/(sum(mc))
err.resub=1-((mc[1,1]+mc[2,2])/sum(mc))
err.resub
