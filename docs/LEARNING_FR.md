# Metamorph : notes d'apprentissage (en français)

Ce fichier rassemble, phase par phase, les concepts clés, les équations et les
questions probables en entretien, avec des éléments de réponse.

---

## Phase 1 : modèle direct

### 1.1 Ce que fait le modèle

On suit une dose injectée par voie intraveineuse (100 %ID) dans le sang et cinq
tissus. Chaque particule est soit nue (b), soit coatée (c). Le passage nu vers
coaté se fait seulement dans le sang, au taux k_coat, et il est irréversible.

États (14) : B_b, B_c, T_{i,b} et T_{i,c} pour les 5 tissus, puis deux
compartiments cumulatifs A_renal (urine) et A_hepatic (bile).

Équations :

```
dB_b/dt       = -k_coat B_b - sum_i k_in[i,b] B_b + sum_i k_out[i] T_i,b - CL_renal B_b
dB_c/dt       =  k_coat B_b - sum_i k_in[i,c] B_c + sum_i k_out[i] T_i,c
dT_i,s/dt     =  k_in[i,s] B_s - k_out[i] T_i,s - [i = foie] CL_liver T_i,s
dA_renal/dt   =  CL_renal B_b
dA_hepatic/dt =  CL_liver (T_foie,b + T_foie,c)
```

### 1.2 Concepts clés

**Modèle compartimental.** Chaque compartiment est supposé homogène (bien
mélangé). Les flux sont du premier ordre : le flux sortant est proportionnel à
la quantité présente. Un taux k (en h^-1) correspond à un temps caractéristique
1/k et à une demi-vie ln2/k.

**Système linéaire.** Toutes les équations sont linéaires en y, donc
dy/dt = A y, avec A une matrice 14 x 14. La solution exacte est
y(t) = expm(A t) y0. On l'utilise comme troisième vérification
(`simulate_exact`).

**Conservation de la masse.** Chaque colonne de A somme à zéro : tout ce qui
quitte un état entre dans un autre. Donc d(somme y)/dt = 1^T A y = 0, et la
somme reste égale à la dose. C'est pour cela qu'on a ajouté A_renal et
A_hepatic : sans eux, le système perdrait de la masse et le test serait moins
simple.

**Valeurs propres et raideur.** Les valeurs propres de A (toutes réelles et
négatives, sauf deux nulles pour les compartiments cumulatifs) donnent les
échelles de temps du système. Le rapport entre la plus rapide et la plus lente
mesure la raideur. Un rapport modéré permet un solveur explicite (Tsit5). Un
rapport très grand imposerait un solveur implicite (Kvaerno5 dans diffrax,
LSODA ou BDF dans SciPy). Les valeurs exactes sont dans
`results/phase1_summary.txt`.

### 1.3 Choix d'implémentation et justification

| Choix | Pourquoi |
|---|---|
| JAX + diffrax | Le solveur est différentiable : NUTS (phase 3) a besoin du gradient de la log-vraisemblance par rapport aux paramètres. |
| Tsit5 à pas adaptatif, rtol 1e-8 | Problème peu raide ; méthode explicite d'ordre 5, efficace. Tolérances serrées pour tester l'accord à 1e-6 près. |
| Référence SciPy écrite en boucle | Deux implémentations indépendantes (une vectorisée, une littérale) : une erreur d'indice dans l'une n'apparaîtrait pas dans l'autre. |
| Solution exacte par expm | Le système est linéaire, donc on a une vérité exacte, pas seulement un accord entre deux solveurs approchés. |
| float64 dans JAX | Le cerveau contient environ 0.3 %ID contre 40 %ID pour le foie ; la précision simple (environ 1e-7 relatif) serait juste pour le test de masse. |
| Quantités en %ID | Pas besoin des volumes d'organes ; directement comparable aux données de biodistribution, souvent exprimées en %ID. |

### 1.4 Tests (tests/test_model.py)

1. Le regroupement des paramètres (`expand_rates`) donne les vecteurs attendus.
2. Les trois formes du second membre (JAX, NumPy en boucle, matrice A)
   coïncident sur des états aléatoires.
3. Les colonnes de A somment à zéro.
4. diffrax, SciPy et la solution exacte coïncident (rtol 1e-6).
5. La masse totale vaut 100 pour les trois méthodes, et aussi pour des
   paramètres tirés au hasard (la conservation est structurelle, pas un hasard
   des valeurs vraies).
6. Les quantités éliminées sont croissantes, les quantités restent positives.
7. Si k_coat = 0, il n'y a aucune particule coatée.

### 1.5 Résultats (source : `scripts/phase1_forward.py`, `results/phase1_summary.txt`)

- Erreur de masse maximale sur 481 instants : 2.8e-14 %ID (diffrax), 7.1e-14 %ID
  (SciPy). C'est l'arrondi machine en float64.
- Erreur maximale par rapport à la solution exacte : 5.3e-8 %ID (diffrax,
  rtol 1e-8) et 7.1e-10 %ID (SciPy, rtol 1e-10).
- Taux de décroissance (opposés des valeurs propres de A, en h^-1) : de 0.0195
  (environ 51 h) à 3.40 (environ 18 min). Rapport de raideur : 174. C'est
  modéré, donc Tsit5 (explicite) convient.
- Le mode le plus rapide, 3.40 h^-1, correspond presque exactement à la sortie
  totale de B_b : k_coat + somme des k_in nus + CL_renal = 1.5 + 1.01 + 0.8 = 3.31.
- À 1 h, 72 % du contenu sanguin est coaté.
- Foie : pic d'environ 42 %ID vers 4 h. Rate : environ 13 %ID. Urine
  cumulée : 30 %ID à 24 h, dont l'essentiel est éliminé dans la première heure.
- Cerveau à 24 h : 0.286 %ID, dont 0.193 nu et 0.093 coaté. La plus grande
  partie du contenu cérébral vient donc des particules nues, entrées tôt, avant
  le coating. C'est une observation importante pour la phase 5 : retarder le
  coating ou accélérer la clairance rénale des particules nues changera la dose
  cérébrale.

Lecture de la figure `figures/phase1_trajectories.png` :

- Sang : B_b chute très vite (voies de sortie cumulées 3.3 h^-1), B_c monte puis
  redescend : c'est la signature du changement d'état.
- Poumon et rein (k_out rapide) : pic transitoire, le contenu suit le sang.
- Foie, rate, cerveau (k_out lent) : accumulation, puis décroissance lente.
- Foie et rate sont dominés par l'état coaté (m_RES = 3), le cerveau par l'état
  nu (m_brain = 0.3).
- Élimination : rénale rapide et précoce (seules les particules nues sont
  filtrées, et elles disparaissent vite du sang), hépatique lente et continue.
  Les deux courbes se croisent vers 17.6 h.

### 1.6 Questions probables en entretien

**Q : Le système est linéaire. Pourquoi utiliser un solveur d'EDO plutôt que
l'exponentielle de matrice ?**
R : Pour ce modèle précis, expm suffirait, et je l'utilise comme vérification.
J'utilise un solveur d'EDO parce que le cadre doit rester valable pour des
extensions non linéaires réalistes : captation saturable par les macrophages
(type Michaelis-Menten), dépendance à la dose, coating réversible. Diffrax donne
aussi des gradients par différentiation automatique, nécessaires pour NUTS.

**Q : Comment savez-vous que votre solveur est correct ?**
R : Trois niveaux : deux implémentations indépendantes du second membre
(vectorisée et en boucle) qui coïncident à 1e-12 ; deux solveurs différents
(Tsit5, LSODA) ; et la solution exacte expm(A t) y0. En plus, la conservation de
la masse est vérifiée pour des paramètres aléatoires.

**Q : Pourquoi k_coat = 1.5 h^-1 alors que la couronne se forme en quelques
secondes ou minutes ?**
R : k_coat est un taux effectif pour un changement d'état fonctionnel : la
maturation de la couronne dure, qui modifie la reconnaissance par les
macrophages et le passage des barrières. L'adsorption initiale (couronne molle)
est plus rapide. La valeur est volontairement choisie pour que les deux états
restent visibles sur l'échelle de temps des mesures. C'est une hypothèse
explicite du README.

**Q : Pourquoi seulement 12 paramètres, et pourquoi ces regroupements ?**
R : Avec un taux d'entrée, un multiplicateur et un taux de sortie par tissu, on
aurait environ 20 paramètres, que des données rares ne contraignent pas. Les
regroupements suivent la physiologie : m_RES commun au foie et à la rate
(macrophages du système réticulo-endothélial) ; k_out_fast pour les tissus
d'échange (poumon, rein) ; k_out_slow pour les tissus qui séquestrent (foie,
rate) et pour l'efflux lent du cerveau.

**Q : Pourquoi la clairance rénale ne s'applique-t-elle qu'aux particules nues ?**
R : La filtration glomérulaire a un seuil de taille d'environ 5 à 6 nm. La
couronne protéique augmente le diamètre hydrodynamique au-delà de ce seuil, donc
une particule coatée n'est plus filtrée. C'est aussi le lien avec la phase 5,
où CL_renal dépend du diamètre.

**Q : Quelle est la limite principale de ce modèle ?**
R : Ce n'est pas un modèle PBPK : pas de volumes d'organes ni de débits
sanguins, chaque taux regroupe perfusion, perméabilité et liaison. Les taux ne
sont donc pas directement transposables entre espèces. Un PBPK serait l'étape
suivante avec de vraies données.

**Q : Que se passe-t-il si un paramètre est négatif pendant l'inférence ?**
R : Le modèle n'a plus de sens physique, et la positivité n'est plus garantie.
C'est pour cela qu'en phase 3 les priors sont log-normaux : on échantillonne le
logarithme des paramètres, qui sont donc toujours positifs.

---

## Phase 2 : expérience synthétique

### 2.1 Plan d'expérience

| Observable | Ce qu'on mesure | Instants (h) | Pourquoi |
|---|---|---|---|
| Sang | B_b + B_c (non séparables) | 5, 15, 30 min, puis 1, 4, 8, 24 h | Prélèvement sériel, non terminal : on peut échantillonner souvent. |
| Urine | A_renal cumulée | mêmes 7 instants | Collecte en cage métabolique, non terminale. |
| 5 tissus | T_i,b + T_i,c | 1, 4, 24 h | Le prélèvement d'organe est terminal : chaque instant coûte un groupe d'animaux (ou de puces). |
| A_hepatic | non observée | | Choix volontaire, pour garder la non-identifiabilité k_out_slow / CL_liver (phase 4). |

3 réplicats indépendants par instant, soit 87 mesures au total (21 sang, 21 urine,
45 tissus). Fichiers : `data/synthetic_data.csv` (format long) et
`data/true_parameters.json` (valeurs vraies, sigma, graine, plan).

### 2.2 Modèle de bruit

y_obs = y_vrai * exp(eps), avec eps ~ N(0, sigma^2), sigma = 0.2.

Pourquoi log-normal multiplicatif :

- les quantités sont positives, et le bruit log-normal le garantit ;
- l'erreur des dosages (fluorescence, ICP-MS, radioactivité) est
  approximativement proportionnelle au signal : coefficient de variation constant,
  environ 20 % ici ;
- les données couvrent plus de 3 ordres de grandeur (cerveau 0.03 %ID, sang
  100 %ID). Un bruit additif serait soit énorme pour le cerveau, soit négligeable
  pour le sang ;
- en échelle log, le bruit devient gaussien et homoscédastique : la
  vraisemblance en phase 3 s'écrit simplement log y_obs ~ N(log y_modèle, sigma^2).

Vérification : l'écart-type empirique des résidus log vaut 0.1955 (vrai : 0.2),
la moyenne +0.0096 (`results/phase2_summary.txt`).

### 2.3 Éviter le « crime inverse »

Les données sont générées avec le solveur SciPy (LSODA), alors que l'inférence
utilisera diffrax (Tsit5). Si on générait et ajustait avec exactement le même
code numérique, l'erreur de discrétisation serait identique des deux côtés et
l'ajustement paraîtrait artificiellement bon. C'est ce qu'on appelle le crime
inverse (« inverse crime »). Ici l'écart entre solveurs est de l'ordre de 1e-8 %ID,
donc l'effet est minime, mais le principe est bon à connaître.

### 2.4 Pourquoi les points précoces identifient k_coat et CL_renal

Les deux paramètres vident B_b : k_coat transforme les particules nues en
particules coatées (qui restent dans le sang), CL_renal les envoie dans l'urine.
Comme B_b décroît au taux k_coat + somme k_in,b + CL_renal = 3.31 h^-1 (demi-vie
d'environ 13 min), tout ce qui distingue les deux paramètres se passe dans la
première heure.

Raisonnement en deux temps, sur l'urine :

1. **Aux temps très courts**, B_b est encore proche de 100 et A_renal(t) est
   environ CL_renal * 100 * t. L'urine précoce mesure donc CL_renal seul, sans
   dépendre de k_coat. Les sensibilités calculées le confirment : à 5 min,
   d log(urine) / d log(CL_renal) = +0.97 et d log(urine) / d log(k_coat) = -0.06.
2. **Aux temps longs**, l'urine atteint un plateau. En négligeant les retours
   des tissus, ce plateau est la fraction des sorties de B_b qui part vers le
   rein : CL_renal / (k_coat + somme k_in,b + CL_renal), soit 0.8 / 3.31 = 24 %
   (le modèle donne 30 %, à cause des particules nues qui reviennent du poumon et
   du rein). Ce plateau ne fixe qu'un rapport : on peut augmenter k_coat et
   CL_renal ensemble sans le changer. À 24 h, les sensibilités valent +0.70 pour
   CL_renal et -0.57 pour k_coat, de signes opposés et de tailles proches.

Donc : l'urine précoce fixe CL_renal, le plateau fixe le rapport, et à eux deux
ils fixent k_coat.

Quantification (figure `figures/phase2_early_sensitivity.png`) : on empile les
sensibilités du sang et de l'urine aux instants de mesure, et on calcule la
corrélation entre la colonne k_coat et la colonne CL_renal. Une corrélation
proche de -1 ou +1 signifie que les deux paramètres ont des effets presque
proportionnels sur les données, donc qu'ils sont difficiles à séparer.

- Avec les 7 instants (dont 5, 15, 30 min) : r = -0.74.
- Avec seulement 1, 4, 8, 24 h : r = -0.945.

Sans les points précoces, les deux paramètres deviennent presque colinéaires.

Remarque : le sang total est presque insensible à k_coat dans la première heure
(sensibilité inférieure à 0.05 en valeur absolue). C'est logique : le coating ne
retire rien du sang, il fait passer les particules d'un état à l'autre dans le
même compartiment. Il n'agit sur le sang qu'indirectement, en changeant les taux
de captation tissulaire. C'est pour cela que l'urine est indispensable.

### 2.5 Concept : sensibilité locale et différentiation automatique

La sensibilité locale normalisée est S = d log y / d log theta : de combien de
pourcents change l'observable quand le paramètre augmente de 1 %. On la calcule
par différentiation automatique à travers le solveur diffrax (`jax.jacrev`, mode
inverse). Le mode direct (`jacfwd`) n'est pas disponible avec l'adjoint par défaut
de diffrax, qui est conçu pour le mode inverse : c'est aussi le mode qu'utilise
NUTS pour le gradient de la log-densité.

La matrice des sensibilités (lignes : mesures ; colonnes : paramètres) est la
base de l'analyse d'identifiabilité locale (matrice d'information de Fisher
approximativement proportionnelle à S^T S / sigma^2). On y reviendra en phase 4
avec le posterior complet, qui est une analyse globale.

### 2.6 Questions probables en entretien

**Q : Pourquoi seulement 3 instants pour les tissus ?**
R : Parce qu'en vrai, chaque instant tissulaire coûte un groupe d'animaux
sacrifiés (ou une puce détruite). Le plan reflète cette contrainte : beaucoup de
points pour le sang et l'urine (non terminaux), peu pour les organes.

**Q : Vos réplicats sont-ils réalistes ?**
R : Partiellement. Je les tire indépendamment. En réalité, les prélèvements
sanguins successifs viennent souvent du même animal, donc ils sont corrélés, et il
existe une variabilité entre individus. Un modèle hiérarchique (paramètres
propres à chaque animal, tirés d'une loi de population) serait l'extension
naturelle.

**Q : Pourquoi un seul sigma pour toutes les observables ?**
R : Pour la simplicité et l'identifiabilité. Avec de vraies données, j'utiliserais
au moins un sigma pour les mesures sanguines et urinaires (erreur de dosage) et
un autre pour les tissus (variabilité biologique plus forte). C'est un changement
d'une ligne dans le modèle NumPyro.

**Q : Pourquoi ne pas observer A_hepatic ?**
R : C'est souvent le cas en pratique (la collecte fécale est plus difficile et
moins quantitative), et cela laisse une non-identifiabilité instructive entre
k_out_slow et CL_liver, étudiée en phase 4.

**Q : Pourquoi les points précoces sont-ils indispensables ?**
R : Voir 2.4. L'urine précoce mesure CL_renal seul, car le coating n'a pas
encore eu le temps d'agir ; le plateau urinaire ne mesure qu'un rapport entre
CL_renal et k_coat. Sans les points à 5, 15 et 30 min, la corrélation entre
leurs sensibilités passe de -0.74 à -0.945.

**Q : Qu'est-ce que le crime inverse ?**
R : Générer les données synthétiques et les ajuster avec exactement le même
modèle numérique. Les erreurs numériques s'annulent et le résultat est trop
optimiste. J'utilise deux solveurs différents pour la génération et l'inférence.
