PFont font; 
String fontFile = "ArialMT-48.vlw"; 
int maxfonte = 230; 
int minfonte = 5; 
Tag[] tags = new Tag[200];
int nTags;
 
void setup(){ 
  size(1500,1000); 
  smooth(); 
  textAlign(LEFT);
  font = loadFont(fontFile);
  // lecture du fichiers des tags
  String lines[] = loadStrings("nuage_mots.txt");
  for (int i=0; i < lines.length; i++) {
   String params[] = split(lines[i]," ");
   tags[i] = new Tag(int(params[2]),params[1],int(params[0]));
  }
  nTags = lines.length;
  // calcul dimensions et position initiale des tags
  int maxpoids = 0; int minpoids = 0;
  for (int i=0; i<nTags; i++) { 
   maxpoids = int(max(maxpoids,tags[i].poids)); 
   minpoids = int(min(minpoids,tags[i].poids));
  }
  for (int i = 0; i < nTags; i++) { 
    tags[i].tf = int(map(tags[i].poids, minpoids, maxpoids, minfonte, maxfonte)); 
    textFont(font, tags[i].tf);
    tags[i].h =  int(textAscent()+textDescent());
    tags[i].w = int(textWidth(tags[i].mot));
    tags[i].bouge(); 
  } 
} 
 
void draw(){
  // recherche des collisions
  int chocs = 0;
  for(int i=0; i<nTags; i++){ 
   for (int j=i+1; j<nTags; j++) {
     if (tags[i].mecoupe(tags[j])) {
       tags[j].bouge();
       chocs++;
       println("choc "+chocs+" moi "+i+" lui "+j);
     }
   }
  }
  // dessins des tags
  background(255);
  for(int i=0; i<nTags; i++){ 
    tags[i].dessin(); 
  } 
  //test d'arret
  println("FINI "+frameCount);
  if (chocs == 0) {saveFrame("resu####.png");noLoop();} else chocs = 0;
} 

class Tag{ 
  int x, y, w, h, poids, tf, qualite; String mot; 
   
  Tag(int laqualite, String lemot, int lepoids){ 
    qualite = laqualite;mot = lemot;poids = lepoids;
    x = y = w = h = tf = 0; 
  } 
  
  void bouge(){
   x = int(random(0,width-w)); 
   y = int(random(h,height)); 
  }
  
  void dessin(){ 
    if (qualite == 0) {fill(255,0,0);textFont(font,tf);text(mot,x,y);}
    else {fill(0,255,0);textFont(font,tf);text(mot,x,y);}
    //noFill();stroke(0);rect(x,y-h,w,h);
   }
   
  boolean mecoupe(Tag l){
   return ! ( l.x > x+w || l.x+l.w < x || l.y-l.h > y || l.y < y-h );
  }    
} 
