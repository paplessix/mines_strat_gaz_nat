# mines_strat_gaz_nat

Projet Ecole des Mines de PARIS en partenariat avec E_Cube : 

- *Sebastian Partarrieu*
- *Pierre-Adrien Plessix*
- *Youri Tchouboukoff*

to install useful Paskages of this project type : `pip install -r setup.py`
## Partie 1 : Web_Scraping
-------------------

Module scrap_saver qui a deux fonctions : initialiser la BDD et updater la BDD. 

Les données sont extraites depuis `https://www.powernext.com/futures-market-data` pour les prix **futures**. Tandis que les prix **Spots** sont issus de `https://www.powernext.com/spot-market-data`. 

Lorsque l'on s'intéresse à la structure générale du site, on comprend mieux comment l'on va s'attaquer à l'extraction des données. On comprend que le contenu est divisé en quatre ou cinq blocs, contenant chacun un graphique et une barre d'onglet permettant d'afficher les données du marché intéressant. 

Cependant le principal problème est que les données ne sont pas stoquées en propre dans le code du site, mais son acquise par une requête d'un script `JavaScript` à une base de donnée extérieure au site. Il est donc nécessaire de se rapprocher le plus possible du comportement utilisateur. 

On utilise donc un module qui permet donc de cliquer sur des boutons. Pour se placer à chaque fois dans les configurations où l'on a accès aux données dans le code HTML. On utilise alors un Parser pour récupérer les données brutes

Dans le cadre de la lecture d'un Graphe, il st nécessaire de pouvoir passer la souris au dessus de la courbe pour avoir accès aux valeurs de prix à cet instant. Encore une fois dans le cas contraire ces données ne sont pas accesible dans le code de la page. On fait donc bouger la souris d'un pas suffisament petit pour passer au dessus de tous les points du graphique. 

Le Package en lui-même est composé de 3 sous-modules :

- `graph_scraper` : C'est itérateur est basé sur un algorithme de lecture des données de type graphes dans une page de Powernext, ce qui dans notre cas correspond à l'extraction de prix _Spots_. Il utilise le module `Selenium` pour bouger la souris sur toute la largeur du graphique. Il itère sur toutes les types de _GNL_ différend. 
    - `Parameters`:
      - Il prend en argument `specific_type`, par défaut `False` , mais on peut ainsi accéder un type de GNL précis.
      - Il prend en argument `link`, le lien auquel chercher le lien le graphique. 
      - Il prend en argument `bloc_number`, le numéro du bloc qui contient le graphe sur la page. Par défaut 0 dans le cadre de la page qui contient les prix spots. 
    - `Return` : 
      - Il retourne de façon itérative, l'ensemble des données de prix contenus sur la page, l'unité ainsi que le type de GNL considéré. 
      - Dans l'ordre `unit, active, table `
    - La fonction lève des exceptions quand : 
      - **LEVER DES EXCEPTIONS**
      - 
- <a id = 'table_scraper'>`table_scraper`</a> :  C'est itérateur est basé sur un algorithme de lecture des données de type tableau dans une page de Powernext, ce qui dans notre cas correspond à l'extraction de prix _Forward_. Il utilise le module `Selenium` pour accéder au code contenu dans le code `HTML`. Il itère sur toutes les types de _GNL_ différents.
    - `Parameters`:
      - Il prend en argument `specific_type`, par défaut `False` , mais on peut ainsi accéder un type de GNL précis.
      - Il prend en argument `link`, le lien auquel chercher le lien le graphique. 
      - Il prend en argument `bloc_number`, le numéro du bloc qui contient le graphe sur la page. Par défaut 3 dans le cadre de la page qui contient les prix forwards.

    - `Return` : 
      - Il retourne de façon itérative, l'ensemble des données de prix contenus sur la page, l'unité ainsi que le type de GNL considéré. 
      - Dans l'ordre `unit, active, table `
    - La fonction lève des exceptions quand : 
      - Top

  > Ces deux modules ont donc une architecture assez similaire, et les méthodes de Scrapping, que ce soit dans le cadre de graphe, ou bien dans le cadre de tableaux, sont intégrés dans une classe `Browser`.
  
  > Ces deux premiers modules reposent sur un driver, `chromedriver.exe`, fournit avec le repo, et disponible à la racine du Repo. Il est nécessaire de posséder Chrome pour faire fonctionner les tâches automatisées. 
- `scrap_saver` : Fonction qui construit et update la base de données à partir des données fournies par <a href= #table_scraper>`table_scraper`</a> et `graph_scraper`, en deux partie, il scrappe d'abord toutes les données du jour, et enfin il ajoute aux données existantes les données qui n'ont pas été encore enregistrées. 
  > On considère ici que il n'y a pas de consolidation à posteriori des données. Faisant que les données précedemment enregistrées ne puissent plus être considérée valables
  - Le Module est lancé par appel dans le terminal de la commande :
`python Web_Scraping/scrap_saver.py -d data_directory -p Product_type -s Specific_Market`
    - Où `-d/--directory` correspond à la position relative du fichier d'enregistrement des données
    - Où `-p/--product` correspond au type de produit dont on désire la mise à jour, si non spécifié les deux, sinon rentrer `'Spot' ou 'Forward'`
    - Où `-s/--specific` correspond au marché désiré. 
    - 'Rajouter des classe pour atteindre des données intéressantes ? Mettre sous forme de classe ? '
  - Le Module retourne l'ensemble des données mises à jour (si elles n'existent pas, le module crée l'architecture de donnée) au format `.csv`  

> Dans notre mise en place, les données les plus récentes sont sockées dans le fichier `./Data`. Où certaines sauvegardes sont conservées

Tests mis en place : 
- 1. 
- 2. 
- 3.
- 
## Partie 2 : Diffusion de Prix 
-------------------


## Partie 3 : Optimisation du stockage
------------------

La première étape consiste à optimiser la gestion d'un stockage de Gaz naturel en supposant que l'on connait parfaitement le prix du Gaz Naturel sur toute une période donnée.

### Modélisation

On va modéliser un Stockage souterrain de Gaz cela correspond à la Classe `Stockage` définie dans `storage_optimisation/stockage.py`.
- Le stockage possède des contraintes intrinsèques que sont essentiellement sont volume et son remplissage au début de la période
- Le Stockage possède des débits d'injection et de soutirage qui varient en fonction du remplissage
- Dans une optique d'assurer des stocks stratégiques, le stockage doit être quasimment plein au début de l'hiver. Et dans une volonté d'assurer la pérennité du stockage, celui-ci doit être quasi-vide au début de l'été. Cela se représente par des fonctions portes que doit éviter le volume de remplissage du stockage

- Les règles dépendent des caractéristiques des  différents stockage, c'ets pourquoi l'on travaillera avec sous-classes, correspondant avec des stockages, ayant des valeurs nominales différentes. Certains sont des stockages à la grande capacité mais très peu flexible sur l'achat-cente, quand d'autres sont beaucoup plus flexibles 

Le stockage est don définit par la classe `Stockage`, qui se construit à partir d'un volume maximal `Vmax`, un volume initial `Vinit`, les prix sur une période `data`, mais aussi la vente algébrique pour chacun de ces jours `evolution`. 

La classe `Stockage` possède les attibuts
- `Vmax`,`data`,`Vinit`,`evolution`
- 

