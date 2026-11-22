<h1 align="center">European AI Championship 2026 · équipe Epi84</h1>

<p align="center">
Trois défis de sécurité de l'IA, 22 heures, un nœud de 4 GPU.<br/>
Paris, 21–22 novembre 2026.
</p>

| | Défi | Ce qu'il fallait faire | Score public à la clôture |
|:-:|---|---|:-:|
| 1 | [Localiser un tatouage invisible](tache1_watermark/) | Trouver, dans chaque image, la zone qui porte un tatouage | **0,762** |
| 2 | [Reconnaître le modèle derrière une API](tache2_model_equality/) | Dire quels points d'accès servent exactement le même modèle | **0,85** |
| 3 | [Faire oublier des images à un modèle](tache3_unlearning/) | Rendre un modèle indiscernable d'un modèle réentraîné sans 3 500 images | **0,324** |

> Ces scores viennent du classement public, calculé sur une partie des tests.
> Le classement final utilise une partie cachée.

---

## 1 · Localiser un tatouage invisible

**Le problème.** Une image porte un tatouage invisible (PixelSeal, TrustMark ou StegaStamp) sur une partie
seulement de sa surface. Il faut rendre une carte de 128 × 128 cases qui dit où il se trouve. Le score est un
mIoU. On dispose de 170 images d'exemple par méthode.

**L'idée.** Fabriquer nous-mêmes les exemples. Les organisateurs fournissent l'outil qui pose le tatouage :
si c'est nous qui tatouons une zone, nous connaissons la réponse exacte, autant de fois que nécessaire.

```mermaid
flowchart LR
    A["Images des deux autres méthodes<br/>(sans le tatouage étudié)"] --> B["Outil officiel de tatouage<br/>+ message aléatoire"]
    B --> C["Zone tatouée connue<br/>= vérité terrain gratuite"]
    C --> D["Réseau de segmentation<br/>encodeur = extracteur officiel"]
    D --> E["Affinage sur les vraies images<br/>validation croisée à 5 plis"]
```

| Étape | Score public |
|---|:-:|
| Réseau en U sur des cartes de corrélation | 0,480 |
| Recherche de rectangles par décodage du message | 0,504 |
| Segmentation apprise sur données synthétiques | 0,654 |
| Entraînement synthétique plus long | 0,725 |
| Plusieurs réseaux moyennés et prédiction en miroir | **0,762** |

Tous les réglages sont choisis en validation croisée sur les images d'entraînement (0,751 en local),
jamais sur le classement. À chaque soumission, le score public est resté à environ 0,02 du score local.

**Où lire.**
[`solution_finale/v61/seg_synth_new.py`](tache1_watermark/solution_finale/v61/seg_synth_new.py) : générateur d'exemples et réseau ·
[`solution_finale/v62/decide62.py`](tache1_watermark/solution_finale/v62/decide62.py) : assemblage final ·
[`RECHERCHE.md`](tache1_watermark/RECHERCHE.md) : les hypothèses testées, y compris celles qui ont été rejetées.

---

## 2 · Reconnaître le modèle derrière une API

**Le problème.** Trente-trois points d'accès anonymes, cent réponses autorisées pour chacun. Il faut dire
lesquels servent le même modèle. Le piège : un déguisement (consigne cachée, persona, tatouage du texte,
température) garde le même modèle mais change beaucoup le texte, alors qu'une quantification ou un
réentraînement léger change le modèle mais presque pas le texte. Comparer les réponses ne suffit donc pas.

**L'idée.** Ne pas demander « qui se ressemble ? » mais « qui porte une preuve de déguisement ? ». Et comme
les modèles candidats sont ouverts, mesurer exactement à quel point chacun aurait pu écrire une réponse.

```mermaid
flowchart LR
    A["Réponses des<br/>33 points d'accès"] --> B["Vraisemblance sous<br/>des modèles ouverts"]
    B --> C["3 familles de 11"]
    C --> D{"Preuve de<br/>déguisement ?"}
    D -- "oui" --> E["Même modèle<br/>que la base"]
    D -- "non" --> F["Reste seul"]
```

| Étape | Score public |
|---|:-:|
| Familles par vraisemblance, variantes par test statistique | 0,40 |
| Preuves de déguisement : réponses identiques, mots rares, fausse identité | 0,70 |
| Consigne cachée récitée par un point d'accès, base d'une famille identifiée | **0,85** |

**La limite.** La base n'est identifiée que pour une famille sur trois. Deux candidats sont restés
indiscernables du modèle de base avec cent réponses chacun : un réentraînement très léger est invisible
à ce budget.

**Où lire.**
[`analyse/decision.py`](tache2_model_equality/analyse/decision.py) et
[`analyse/temperature.py`](tache2_model_equality/analyse/temperature.py) : familles et test de vraisemblance ·
[`solution_finale/v80/NOTES.md`](tache2_model_equality/solution_finale/v80/NOTES.md) : la dernière matinée, test par test.

---

## 3 · Faire oublier des images à un modèle

**Le problème.** Un classifieur d'images (ViT-B/16, 100 classes) doit oublier 3 500 images d'entraînement.
Le juge est un modèle réentraîné sans elles : il faut lui ressembler image par image, sans le posséder,
sans les autres images d'entraînement et sans les étiquettes. Le score vaut zéro si le modèle perd trop
en précision.

**L'idée.** Oublier, ce n'est pas tout rater. Un élève qui a appris des corrigés par cœur doit, pour faire
comme s'il ne les avait jamais vus, hésiter là où un autre élève hésiterait.

```mermaid
flowchart LR
    A["Modèle d'origine<br/>(professeur)"] --> B["Cible : moyenne sur 16 vues,<br/>bonne classe abaissée"]
    B --> C["L'élève imite la cible<br/>sur les images à oublier"]
    A --> D["Sur des photos extérieures,<br/>l'élève imite le professeur"]
    C --> E["Seules les images fragiles<br/>sont poussées vers l'erreur"]
    D --> E
```

| Étape | Score public |
|---|:-:|
| Première cible adoucie | 0,014 |
| Oublier davantage | 0,045 |
| Éteindre les classes presque entièrement retirées | 0,135 |
| Cible moyennée sur 16 vues de chaque image | 0,151 |
| N'éteindre que deux classes | 0,246 |
| Viser des images fragiles plutôt que des classes | 0,286 |
| Deux indicateurs de fragilité combinés | 0,294 |
| Tout rendre un peu plus hésitant | 0,314 |
| Réglage de la marge de l'étape 2 | **0,324** |

**La limite.** Environ 75 soumissions ont servi à régler la méthode sur le classement public : le risque de
trop coller à ce classement est réel.

**Où lire.**
[`JOURNAL.md`](tache3_unlearning/JOURNAL.md) : la démarche, version après version ·
[`versions/v47/`](tache3_unlearning/versions/v47/) : la recette en deux étapes ·
[`versions/v70/NOTES.md`](tache3_unlearning/versions/v70/NOTES.md) : les derniers réglages.

---

## Ce qu'on retient

- Quand on a l'outil officiel entre les mains, on peut fabriquer ses propres données d'entraînement.
- Quand les modèles sont ouverts, on peut tester exactement, et connaître la limite du test.
- Avant d'optimiser, comprendre ce que la métrique récompense.

## Organisation du dépôt

```
tache1_watermark/         énoncé, journal de recherche, scripts de recherche, solution finale (v60 à v63)
tache2_model_equality/    énoncé, scripts d'analyse, solution finale (v80)
tache3_unlearning/        énoncé, journal, baseline, une version par idée (versions/v1 à v70)
infrastructure_cluster/   files d'attente des GPU, scripts de soumission, journal des scores
```

Chaque dossier de version contient son code, ses scripts de lancement et un `NOTES.md` : l'idée, les
paramètres, les mesures locales et le score obtenu.

## Ce qui n'est pas ici

Les poids, les données, les réponses des API et les journaux d'exécution sont restés sur le cluster.
Les chemins et identifiants du cluster sont anonymisés dans les scripts : il faut les rétablir pour les
relancer.
