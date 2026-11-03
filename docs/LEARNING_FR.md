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

---

## Phase 3 : calibration bayésienne

### 3.1 Le modèle probabiliste

```
z_j     ~ Normal(0, 1)                       j = 1..12
theta_j = exp(log(mediane_j) + s_j * z_j)    donc theta_j ~ LogNormal(log mediane_j, s_j)
sigma   ~ HalfNormal(0.5)
log y_obs,n ~ Normal(log y_modele,n(theta), sigma^2)
```

y_modele est obtenu en résolvant l'EDO avec diffrax à chaque évaluation de la
log-densité. NUTS a besoin du gradient de cette log-densité par rapport aux z_j :
JAX le calcule par différentiation automatique à travers le solveur (mode
inverse).

Théorème de Bayes : p(theta | données) est proportionnel à
p(données | theta) * p(theta). NUTS ne calcule jamais la constante de
normalisation ; il n'a besoin que de la log-densité non normalisée et de son
gradient.

### 3.2 Les priors : raisonnement par ordres de grandeur

| Paramètre | Médiane | s (log) | Raisonnement |
|---|---|---|---|
| k_coat | 1 /h | 1.0 | Maturation de la couronne dure en minutes à heures. |
| CL_renal | 0.5 /h | 1.0 | Demi-vie sanguine de dizaines de minutes à quelques heures pour des particules ultrapetites. |
| CL_liver | 0.1 /h | 1.0 | Élimination hépatobiliaire lente (heures à jours). |
| k_in (poumon, foie, rate, rein) | 0.2 /h | 1.0 | Captation par des organes bien perfusés en quelques heures. |
| k_in_brain | 0.01 /h | 1.0 | Barrière hémato-encéphalique : 1 à 2 ordres de grandeur en dessous des autres organes. |
| m_RES | 2 | 0.7 | Opsonisation : captation macrophagique attendue plus forte, mais une baisse n'est pas exclue. |
| m_brain | 0.5 | 0.7 | La couronne gêne le passage de la barrière, mais une hausse n'est pas exclue. |
| k_out_fast | 0.3 /h | 1.0 | Retour des tissus d'échange en quelques heures. |
| k_out_slow | 0.05 /h | 1.0 | Séquestration : retour en environ une journée. |
| sigma | HalfNormal(0.5) | | Coefficient de variation probablement sous 100 %. |

Points à retenir :

- Les médianes ne sont pas les valeurs vraies, qu'un modélisateur ne connaît
  pas. Plusieurs sont décalées d'un facteur 2 (k_in_liver 0.2 contre 0.4,
  k_in_spleen 0.2 contre 0.1, CL_liver 0.1 contre 0.05).
- s = 1 veut dire qu'un écart-type correspond à un facteur e = 2.7, et que
  l'intervalle à 90 % du prior couvre un facteur exp(2 * 1.645) = 27, soit plus
  d'un ordre de grandeur. Ce sont des priors faiblement informatifs.
- Pourquoi log-normal : les taux sont positifs et on raisonne en ordres de
  grandeur (facteurs multiplicatifs), pas en écarts additifs.

### 3.3 Vérification prédictive a priori (prior predictive check)

On tire 1000 jeux de paramètres dans le prior, on simule, on ajoute le bruit, et
on compare aux données avant tout ajustement. Le but est de vérifier que le
prior produit des courbes plausibles, ni absurdes, ni trop restrictives.

Résultats (`results/phase3_summary.txt`) :

- 0 échec du solveur sur 1000 tirages ;
- 98.9 % des points de données sont dans la bande prédictive à 90 % du prior :
  le prior ne les exclut pas ;
- cerveau à 24 h : intervalle à 90 % du prior [0.034, 2.7] %ID, ordre de
  grandeur plausible et prior large ;
- urine à 24 h : [3.8, 71] %ID, large mais physiquement possible.

Remarque : les données du foie sont dans la partie haute de la bande, car la
médiane du prior pour k_in_liver (0.2) est la moitié de la vraie valeur (0.4).
Les données doivent corriger le prior, et c'est ce qu'elles font.

### 3.4 Paramétrisation et diagnostics : problèmes rencontrés et solutions

**Paramétrisation en log standardisé (non centrée).** Les taux couvrent trois
ordres de grandeur (0.01 à 1.5 /h) et doivent rester positifs. On échantillonne
z_j ~ N(0, 1), sans contrainte et d'échelle 1, puis
theta_j = exp(mu_j + s_j z_j). Avantages : pas de bord à theta = 0, des échelles
comparables pour tous les paramètres, et des corrélations multiplicatives qui
deviennent linéaires en log.

**Problème 1 : les chaînes parallèles ne fonctionnaient pas.** Avec jax 0.10 et
diffrax 0.7, le mode `parallel` de NumPyro (pmap, une chaîne par cœur) échoue
dans diffrax avec une erreur de « closure conversion ». J'ai vérifié que l'erreur
ne vient pas de l'option `throw=False`. Solution : chaînes séquentielles. Les
résultats sont les mêmes en loi, seul le temps de calcul change.

**Problème 2 : échantillonnage lent, avec une cause mesurable.** Un premier essai
court prenait environ 50 gradients par itération. Un gradient coûte environ 8 ms,
donc le coût venait du nombre de pas leapfrog, c'est-à-dire de la géométrie du
posterior. Le posterior a de fortes corrélations linéaires en espace log
(k_coat avec CL_renal, k_in_liver avec m_RES), et une matrice de masse diagonale
ne peut pas les absorber : NUTS doit faire beaucoup de petits pas. Mesure
(`scripts/phase3_mass_matrix.py`, une chaîne, même graine) :

| Matrice de masse | Pas leapfrog / itération | ESS min | ESS pour 1000 gradients | Divergences |
|---|---|---|---|---|
| diagonale | 58.3 | 97 | 5.55 | 0 |
| dense | 8.6 | 201 | 78.15 | 0 |

La matrice de masse dense (13 x 13, adaptée pendant le warmup) est environ 14 fois
plus efficace. Elle revient à un changement de variables linéaire qui
« blanchit » le posterior.

**Problème 3 : paramètres extrêmes pendant le warmup.** Au début du warmup, NUTS
peut proposer des taux très grands. Le solveur est appelé avec `throw=False` :
s'il échoue, il renvoie des valeurs non finies, la log-densité est non finie, et
NumPyro rejette la proposition au lieu d'arrêter le calcul. Aucune divergence
n'a été observée après le warmup.

### 3.5 Diagnostics de l'exécution finale

4 chaînes x (1000 warmup + 1000 tirages), matrice de masse dense,
target_accept 0.8, 609 s au total.

- **Divergences** : 0 / 4000. Une divergence signale une région de forte
  courbure que l'intégrateur leapfrog n'arrive pas à suivre ; les tirages
  seraient alors biaisés.
- **R-hat** max : 1.000 (seuil usuel 1.01). R-hat compare la variance entre
  chaînes et la variance dans chaque chaîne. Proche de 1, il indique que les 4
  chaînes, parties de points différents, explorent la même distribution.
- **ESS** min : 2450 (bulk) et 1948 (tail), sur 4000 tirages. L'ESS est le
  nombre de tirages indépendants équivalents, compte tenu de l'autocorrélation.
  L'ESS bulk concerne la moyenne, l'ESS tail les quantiles à 5 % et 95 %.
- Pas leapfrog moyens par itération : 7.8.

### 3.6 Vérification prédictive a posteriori

93.1 % des points de données sont dans la bande prédictive à 90 % du posterior
(attendu : environ 90 %). La figure `figures/phase3_posterior_predictive.png`
montre deux bandes :

- bande foncée : incertitude sur la courbe du modèle (paramètres seuls) ;
- bande claire : bande prédictive, qui ajoute le bruit d'observation. C'est
  celle qu'on compare aux points de données.

### 3.7 Résultats : valeurs vraies et posterior

| Paramètre | Vrai | Moyenne post. | IC 90 % | Largeur (haut / bas) |
|---|---|---|---|---|
| k_coat | 1.5 | 1.92 | [1.39, 2.51] | 1.81 |
| CL_renal | 0.8 | 0.921 | [0.785, 1.07] | 1.37 |
| CL_liver | 0.05 | 0.0529 | [0.0438, 0.0625] | 1.43 |
| k_in_lung | 0.3 | 0.256 | [0.207, 0.313] | 1.51 |
| k_in_liver | 0.4 | 0.338 | [0.177, 0.54] | 3.05 |
| k_in_spleen | 0.1 | 0.087 | [0.045, 0.139] | 3.08 |
| k_in_kidney | 0.2 | 0.166 | [0.134, 0.205] | 1.53 |
| k_in_brain | 0.01 | 0.0118 | [0.0085, 0.0153] | 1.80 |
| m_RES | 3 | 4.10 | [1.99, 7.50] | 3.77 |
| m_brain | 0.3 | 0.373 | [0.131, 0.775] | 5.92 |
| k_out_fast | 0.5 | 0.434 | [0.350, 0.535] | 1.53 |
| k_out_slow | 0.03 | 0.0349 | [0.0273, 0.0437] | 1.60 |
| sigma | 0.2 | 0.196 | [0.171, 0.224] | 1.31 |

Les 13 valeurs vraies sont dans leur IC à 90 %. Avec 13 paramètres, on
s'attendrait en moyenne à environ 1.3 valeur hors de l'intervalle ; 13 sur 13
n'est pas suspect, d'autant que les paramètres sont corrélés et ne constituent
donc pas 13 épreuves indépendantes.

Lecture :

- **Bien déterminés** (facteur inférieur à 2) : sigma, CL_renal, CL_liver, les
  k_in du poumon, du rein et du cerveau, k_out_fast, k_out_slow, k_coat. Grâce
  aux points précoces, k_coat et CL_renal sont tous les deux bien contraints
  (phase 2).
- **Mal déterminés** : m_brain (facteur 5.9), m_RES (3.8), k_in_liver et
  k_in_spleen (environ 3). Cause attendue : on n'observe que le total nu + coaté.
  Pour le foie, la captation vaut k_in_liver * (B_b + m_RES * B_c). On peut
  augmenter m_RES et diminuer k_in_liver sans presque rien changer au total, d'où
  une corrélation négative entre les deux. Ce sera l'objet de la phase 4.
- **Une surprise utile** : la non-identifiabilité attendue entre k_out_slow et
  CL_liver est faible (facteurs 1.6 et 1.4). Raison probable : k_out_slow est
  partagé avec la rate et le cerveau, qui n'ont pas de clairance. Leur
  décroissance lente fixe k_out_slow, et le foie fixe ensuite CL_liver. Le
  regroupement des paramètres agit donc aussi sur l'identifiabilité. À vérifier
  en phase 4 sur les corrélations.
- k_coat est estimé un peu haut (1.92 contre 1.5), et CL_renal aussi (0.92 contre
  0.8). Les deux écarts vont dans le même sens, ce qui est cohérent avec le
  rapport CL_renal / (k_coat + ...) fixé par le plateau urinaire (phase 2). C'est
  un effet du bruit sur ce jeu de données précis, et la vraie valeur reste dans
  l'IC.

### 3.8 Questions probables en entretien

**Q : Pourquoi NUTS et pas Metropolis-Hastings ?**
R : En dimension 13 avec des corrélations, une marche aléatoire est très
inefficace. HMC utilise le gradient pour faire de longs déplacements dans des
directions de forte probabilité. NUTS règle automatiquement la longueur des
trajectoires (critère « no U-turn »). Le gradient est disponible grâce à la
différentiation automatique à travers diffrax.

**Q : Comment savez-vous que vos chaînes ont convergé ?**
R : R-hat = 1.000 sur tous les paramètres avec 4 chaînes, ESS supérieur à 1900,
aucune divergence, et les traces se superposent. Ces diagnostics ne prouvent
pas la convergence, mais ils détectent les échecs les plus courants.

**Q : Que feriez-vous en cas de divergences ?**
R : D'abord augmenter target_accept (0.9 ou 0.95), ce qui réduit le pas
d'intégration. Si elles persistent, c'est un problème de géométrie : il faut
reparamétrer (log, non centré), revoir des priors trop larges qui laissent
entrer des régions raides, ou corriger un modèle mal identifié. Il ne faut jamais
ignorer les divergences.

**Q : Pourquoi une matrice de masse dense ?**
R : Je l'ai mesuré : les pas leapfrog par itération passent de 58 à 8.6, et
l'efficacité est multipliée par 14. Le posterior a des corrélations linéaires
fortes en log, qu'une matrice dense absorbe. En dimension 13, son coût est
négligeable. En grande dimension, je préférerais une reparamétrisation
explicite ou une structure de bas rang.

**Q : Pourquoi la vérification a priori est-elle importante ?**
R : Elle vérifie que le prior encode des connaissances raisonnables : il ne doit
pas exclure les données, ni produire des comportements impossibles qui
gaspilleraient le temps de calcul ou biaiseraient le posterior là où les
données sont peu informatives. C'est le cas de m_brain, dont le posterior reste
proche du prior.

**Q : Votre posterior dépend-il du prior ?**
R : Pour les paramètres bien déterminés, très peu : le posterior est beaucoup
plus étroit que le prior (figure des marginales). Pour m_brain, oui : son
posterior ressemble encore au prior, un peu déplacé. Une analyse de sensibilité
au prior (par exemple s = 0.7 contre 1.5) serait la bonne façon de le montrer.

**Q : Pourquoi un solveur moins précis pour l'inférence ?**
R : Le bruit des données est de 20 %. Une tolérance relative de 1e-6 donne une
erreur numérique des centaines de milliers de fois plus petite que le bruit, et
accélère chaque évaluation. La précision du solveur doit être adaptée à
l'information contenue dans les données.

**Q : Que représentent les deux bandes de la figure prédictive ?**
R : La bande foncée est l'incertitude épistémique sur la courbe moyenne, due aux
paramètres. La bande claire ajoute le bruit d'observation : c'est ce qu'on
s'attend à mesurer lors d'une nouvelle expérience. Les données doivent se
trouver dans la bande claire, pas forcément dans la foncée.

---

## Phase 4 : identifiabilité et sensibilité

Tous les chiffres viennent de `results/phase4_summary.txt`
(`scripts/phase4_identifiability.py`).

### 4.1 Deux notions à ne pas confondre

- **Identifiabilité structurelle** : avec des données parfaites (sans bruit, en
  continu), peut-on retrouver un paramètre de façon unique ? C'est une propriété
  du modèle et de ce qu'on observe.
- **Identifiabilité pratique** : avec nos données réelles (rares, bruitées), le
  paramètre est-il bien contraint ? C'est ce qu'on lit dans le posterior. Ici,
  tous les paramètres sont probablement identifiables structurellement, mais
  certains le sont mal en pratique.

En bayésien, un paramètre mal identifié n'empêche pas l'inférence : le posterior
reste simplement large, ou suit une « crête » (une combinaison de paramètres
bien déterminée, les autres directions libres). Le prior évite que le posterior
soit impropre.

### 4.2 Corrélations du posterior (en espace log)

| Paire | r | Explication |
|---|---|---|
| k_in_liver, m_RES | -0.97 | Captation hépatique = k_in_liver (B_b + m_RES B_c). Après 30 min, presque tout le sang est coaté, donc le foie ne voit que le produit k_in_liver * m_RES. Les tissus ne sont prélevés qu'à partir de 1 h : le produit est connu, pas chaque facteur. En log, log k_in_liver + log m_RES = constante, d'où une crête de pente -1. |
| k_in_liver, k_in_spleen | +0.97 | Conséquence du partage de m_RES : si m_RES monte, k_in_liver et k_in_spleen doivent baisser ensemble. |
| k_in_spleen, m_RES | -0.95 | Même produit que pour le foie. |
| k_in_brain, m_brain | -0.84 | Même logique de produit, mais plus faible : 2/3 du contenu cérébral est nu, ce qui fixe en partie k_in_brain seul. |
| k_coat, CL_renal | +0.79 | Le plateau urinaire fixe le rapport CL_renal / (k_coat + ...) (phase 2). Les points précoces empêchent la crête d'être parfaite. |
| k_in_lung ou k_in_kidney, k_out_fast | +0.79 | Tissus d'échange : ils atteignent vite un quasi-équilibre avec le sang, T = (k_in / k_out) B. Seul le rapport (coefficient de partage) est bien déterminé. |
| k_out_slow, CL_liver | +0.29 | Faible. Voir ci-dessous. |

**Une prédiction de la phase 1 qui s'est révélée fausse.** J'avais annoncé une
forte non-identifiabilité entre k_out_slow et CL_liver, car le contenu du foie
dépend surtout de leur somme. La corrélation n'est que de +0.29, et de signe
opposé à celui attendu (une somme fixée donnerait une corrélation négative).
L'explication : k_out_slow est partagé avec la rate et le cerveau, qui n'ont pas
de clairance. Leur décroissance lente fixe k_out_slow, et le foie fixe ensuite
CL_liver. Le regroupement des paramètres, choisi pour réduire leur nombre, a donc
aussi supprimé cette non-identifiabilité. C'est bon à dire en entretien : on a
testé une hypothèse et on l'a corrigée au vu des résultats.

### 4.3 Contraction : combien les données ont appris

Contraction = 1 - var_posterior / var_prior (en log). 1 : les données ont tout
appris ; 0 : le posterior est égal au prior.

- Plus de 0.96 : CL_renal, CL_liver, k_in_lung, k_in_kidney, k_out_fast,
  k_out_slow, k_coat, k_in_brain.
- 0.88 : k_in_liver, k_in_spleen.
- 0.67 : m_RES.
- 0.40 : m_brain. C'est le paramètre le moins appris : son posterior reste
  proche du prior.

### 4.4 Quelles données régleraient le problème ?

**Méthode.** Relancer NUTS pour chaque plan d'expérience candidat coûterait
10 min par plan. J'utilise à la place l'approximation de Laplace :

```
J = d log y / d z        (sensibilités, par différentiation automatique)
Information de Fisher I = n_rep * J^T J / sigma^2
Précision du posterior en z ≈ I + Identité   (le prior en z est N(0, I))
Covariance ≈ (I + Identité)^-1
```

C'est l'approximation gaussienne du posterior autour de son mode. Elle est
locale : elle suppose que le modèle est presque linéaire sur la largeur du
posterior.

**Validation.** Sur le plan réel, l'écart-type de Laplace et celui de NUTS
diffèrent de 4 % en médiane et de 10 % au maximum (k_in_spleen). Laplace est un
peu pessimiste pour k_in_liver, k_in_spleen et m_RES (crêtes légèrement
courbées). L'approximation est donc assez bonne pour comparer des plans.

**Résultats** (facteur de largeur de l'IC à 90 %, borne haute / borne basse) :

| Paramètre | Actuel | + fèces | + sang nu/coaté séparé | + tissus précoces | + bras pré-coaté |
|---|---|---|---|---|---|
| k_coat | 1.84 | 1.65 | **1.12** | 1.65 | 1.78 |
| k_in_liver | 3.46 | 1.62 | 1.60 | **1.45** | 2.80 |
| m_RES | 4.27 | 1.89 | 1.84 | **1.72** | 2.89 |
| m_brain | 6.10 | 5.85 | 5.67 | 3.08 | **1.64** |
| CL_liver | 1.44 | **1.23** | 1.41 | 1.41 | 1.28 |

Les plans :

- **+ fèces** : élimination biliaire cumulée (A_hepatic) aux 7 instants denses ;
- **+ sang nu/coaté séparé** : mesurer B_b et B_c séparément (par exemple en
  isolant les particules couvertes de protéines) ;
- **+ tissus précoces** : prélever aussi les organes à 5, 15 et 30 min (trois
  groupes d'animaux en plus) ;
- **+ bras pré-coaté** : un second groupe reçoit des particules pré-incubées dans
  du plasma (100 % coatées), avec les mêmes prélèvements (sans urine, qui serait
  vide).

Lecture :

1. **m_RES et k_in_liver** : ce qui manque, c'est la captation par le foie
   **avant** le coating. Les tissus précoces donnent cette information
   directement. Les fèces la donnent indirectement : les fèces précoces
   mesurent CL_liver * (contenu hépatique précoce). J'ai ajouté le plan
   « tissus précoces » précisément pour tester cette explication. Il donne le
   même gain que les fèces, ce qui la confirme.
2. **Le bras pré-coaté aide peu pour le foie** (3.46 vers 2.80). Il mesure très
   bien le produit k_in_liver * m_RES, mais ce produit était déjà connu. Pour
   séparer les deux facteurs, il faut la captation à l'état nu.
3. **m_brain** : seul le bras pré-coaté le fixe vraiment (6.1 vers 1.64). Le
   contenu cérébral est surtout nu, donc la part coatée est un petit signal dans
   le bruit. Injecter des particules déjà coatées mesure directement
   k_in_brain * m_brain, et le bras nu fixe k_in_brain.
4. **k_coat** : séparer nu et coaté dans le sang observe directement la
   conversion, et le facteur passe de 1.84 à 1.12.

Message général : un paramètre est mal déterminé quand il n'agit qu'à travers un
produit ou un rapport avec un autre. La bonne expérience est celle qui fait
varier un facteur sans l'autre (un état sans l'autre, ou un instant où un seul
mécanisme agit).

### 4.5 Sensibilité globale : qu'est-ce qui décide où la particule finit ?

**Question.** Quels paramètres expliquent la variation des quantités dans le
cerveau et dans le foie à 24 h ?

**Méthode : indices de Sobol (basés sur la variance).** Pour une sortie Y et des
entrées indépendantes X_i :

```
S1_i = Var(E[Y | X_i]) / Var(Y)        effet de X_i seul (premier ordre)
ST_i = E[Var(Y | X_-i)] / Var(Y)       effet total de X_i, interactions comprises
```

S1_i est la part de variance qu'on supprimerait en fixant X_i. ST_i est la part
qui resterait si on fixait tout sauf X_i. ST_i - S1_i mesure les interactions.
Somme des S1 égale à 1 : modèle additif.

Estimateurs : Saltelli (2010) pour S1, Jansen (1999) pour ST, avec une suite de
Sobol quasi aléatoire brouillée, N = 4096 échantillons de base, soit
N (d + 2) = 57 344 simulations, et des intervalles bootstrap à 90 %. Testé sur
deux fonctions dont les indices sont connus analytiquement
(`tests/test_sensitivity.py`).

**Choix à justifier :**

- **Entrées = le prior** (log-normales indépendantes). Sobol suppose des entrées
  indépendantes. Le posterior est corrélé (section 4.2), donc les indices de
  Sobol classiques n'y sont pas valides. Et la question porte sur les
  mécanismes, sur toute la gamme plausible des particules : le prior est la
  bonne population.
- **Sortie = log10 de la quantité.** Les quantités varient sur plusieurs ordres
  de grandeur. En échelle linéaire, la variance serait dominée par quelques
  tirages extrêmes.
- **Pourquoi pas seulement une étude un par un (OAT) ?** Elle est locale (un seul
  point), ignore les interactions, et ne tient pas compte de la gamme plausible
  de chaque paramètre. Je l'ai calculée pour comparer : élasticités locales
  d log(sortie) / d log(theta) au point moyen du posterior.

**Résultats, cerveau à 24 h** (somme des S1 = 0.92, donc presque additif) :

| Paramètre | S1 | ST |
|---|---|---|
| k_in_brain | 0.50 | 0.50 |
| k_out_slow | 0.17 | 0.19 |
| m_brain | 0.07 | 0.09 |
| k_in_liver | 0.06 | 0.08 |
| CL_renal | 0.06 | 0.08 |
| m_RES | 0.04 | 0.06 |
| k_coat | 0.00 | 0.03 |

**Résultats, foie à 24 h** (somme des S1 = 0.88) : CL_liver (S1 0.57, ST 0.64),
k_in_liver (0.11 ; 0.18), CL_renal (0.08 ; 0.10), k_in_spleen (0.05 ; 0.08),
k_coat (0.04 ; 0.06).

**Interprétation : ce qui décide où finit la particule.**

- **Cerveau** : d'abord la perméabilité de la barrière (k_in_brain, 50 % de la
  variance) et la rétention dans le cerveau (k_out_slow, 19 %). Ensuite la
  compétition avec les « puits » qui vident le sang : captation par le foie et
  la rate (k_in_liver, m_RES) et filtration rénale (CL_renal). Tout ce qui retire
  la particule du sang plus vite réduit l'exposition du cerveau. Enfin l'effet
  du coating sur la barrière (m_brain).
- **k_coat a un S1 nul mais un ST de 0.03** : il n'agit que par interaction.
  C'est logique : l'effet d'un coating plus rapide sur le cerveau dépend de
  m_brain. Si m_brain vaut 1, le coating ne change rien pour le cerveau ; s'il
  est petit, coater plus vite protège le cerveau. Le signe de l'effet dépend
  d'un autre paramètre.
- **Foie** : la quantité à 24 h dépend surtout de la vitesse à laquelle le foie
  élimine (CL_liver), puis de la captation et de la compétition avec le rein et
  la rate.

**Local contre global.** Le classement des principaux paramètres est
semblable, mais pas identique. Exemple : m_RES et k_in_liver ont la même
élasticité locale sur le cerveau (-0.41), mais m_RES a un ST plus faible (0.06
contre 0.08), car son prior est plus étroit (s = 0.7 contre 1). Pour un modèle
presque linéaire en log, S_i est environ (e_i s_i)^2 / somme_j (e_j s_j)^2, où
e_i est l'élasticité et s_i l'écart-type log du prior. Une élasticité forte
compte peu si le paramètre varie peu en pratique.

**Conséquence pour la phase 5.** Le diamètre d agit sur CL_renal et sur
k_in_brain, deux leviers importants pour le cerveau. La propension au coating h
agit sur k_coat, qui a un effet global faible et seulement par interaction. On
peut donc prévoir que d sera un levier bien plus efficace que h pour réduire
l'accumulation cérébrale.

### 4.6 Questions probables en entretien

**Q : Votre modèle est-il identifiable ?**
R : En pratique, partiellement. Huit paramètres sur douze sont bien déterminés
(contraction supérieure à 0.96). Quatre le sont mal : m_brain, m_RES,
k_in_liver, k_in_spleen. La raison est structurelle dans le plan d'expérience :
on observe des totaux nu + coaté, et les tissus ne sont prélevés qu'après que la
plupart des particules sont coatées. Seuls les produits k_in * m sont alors
déterminés.

**Q : Quelle expérience proposeriez-vous ?**
R : Cela dépend du paramètre visé. Pour m_RES : des prélèvements tissulaires
précoces (ou la collecte des fèces). Pour m_brain : un second groupe recevant
des particules pré-incubées dans du plasma. Pour k_coat : séparer nu et coaté
dans le sang. Je l'ai quantifié par l'information de Fisher, validée contre NUTS
(écart médian de 4 %).

**Q : Pourquoi ne pas relancer NUTS pour chaque plan ?**
R : 10 minutes par plan contre quelques secondes avec Laplace, qui reproduit
NUTS à 4 % près sur le plan réel. Pour choisir définitivement un plan, je
validerais le meilleur candidat par une vraie inférence sur données simulées.
C'est le principe de l'optimisation bayésienne de plans d'expérience.

**Q : Pourquoi Sobol et pas une étude un par un ?**
R : Une étude un par un est locale et ignore les interactions. Ici, c'est
important : k_coat n'a aucun effet de premier ordre sur le cerveau, mais un effet
total non nul, uniquement par interaction avec m_brain. Une étude un par un au
mauvais point aurait pu conclure que le coating ne compte pas.

**Q : Pourquoi faire Sobol sur le prior et pas sur le posterior ?**
R : Pour deux raisons. Sobol suppose des entrées indépendantes, or le posterior
est fortement corrélé (r = -0.97). Et la question est : quelles propriétés,
parmi toutes les particules plausibles, décident de la destination ? Pour la
question « quelle incertitude restante compte le plus pour une prédiction », il
faudrait des indices adaptés aux entrées corrélées (indices de Shapley, par
exemple).

**Q : Que signifie un S1 négatif ?**
R : Le vrai S1 est positif ou nul. Un estimateur à échantillon fini peut donner
une petite valeur négative quand l'indice vrai est proche de 0. Les intervalles
bootstrap (par exemple [-0.012, +0.004] pour k_coat) le montrent.

**Q : Que vous apprend la corrélation k_in_lung / k_out_fast ?**
R : Pour les tissus d'échange, seul le coefficient de partage k_in / k_out est
bien déterminé, car ils sont presque à l'équilibre avec le sang. C'est un
résultat classique en pharmacocinétique. Le séparer demande des points pendant
la phase de montée, c'est-à-dire avant 1 h.

---

## Phase 5 : design contrefactuel

Tous les chiffres viennent de `results/phase5_summary.txt`
(`scripts/phase5_counterfactual.py`).

### 5.1 La question

La particule de référence (diamètre d = 5 nm, propension au coating h = 1) a
une quantité cérébrale à 24 h supérieure au seuil de 0.2 %ID : 0.294 %ID à la
moyenne du posterior, IC à 90 % [0.256, 0.351]. Quel est le plus petit
changement de (d, h) qui la fait passer sous le seuil ?

Il s'agit d'un **contrefactuel interventionnel** : « que se serait-il passé si
la particule avait eu d'autres propriétés ? ». On répond avec un modèle
mécaniste calibré, en propageant l'incertitude du posterior, et non avec une
corrélation dans des données.

### 5.2 Lois propriétés vers taux (`properties.py`)

```
CL_renal(d)   = CL_renal   * sigmoid((6 - d) / 0.7) / sigmoid((6 - 5) / 0.7)
k_in_brain(d) = k_in_brain * exp(-(d - 5) / 4)
k_coat(h)     = k_coat     * h
```

Chaque loi vaut 1 à la particule de référence. On l'applique à **chaque tirage**
du posterior : l'incertitude sur les taux de la particule de référence se
propage directement à la nouvelle particule. Hypothèse forte, écrite dans le
README : les constantes des lois (6 nm, 0.7 nm, 4 nm) sont supposées connues,
et rien d'autre ne dépend de d ou de h.

Effet de d sur le cerveau, avec deux mécanismes opposés :

- d plus grand, moins de filtration rénale, la particule circule plus
  longtemps : **plus** d'exposition du cerveau ;
- d plus grand, moins de passage de la barrière : **moins** de captation
  cérébrale.

Pour h faible, la quantité cérébrale passe par un minimum vers 6 nm. Le
problème n'est donc pas convexe, d'où la recherche sur grille avant
l'optimisation locale.

### 5.3 Distance normalisée

u = (d - 5) / 2 nm et v = log2(h). Distance = sqrt(u^2 + v^2). Une unité
correspond à 2 nm, ou à un doublement de h. **Le résultat dépend de ce choix** :
si changer h coûtait moins cher en pratique, la solution irait davantage vers h.
En vrai, il faudrait le fixer avec des chimistes (quels changements sont faciles
à fabriquer).

### 5.4 Méthode

1. **Point estimé** : minimiser u^2 + v^2 sous la contrainte
   cerveau(theta_moyen; d, h) <= 0.2. Contrainte écrite en log
   (log 0.2 - log cerveau >= 0, mieux conditionnée). Recherche sur grille
   41 x 41, puis SLSQP avec gradients JAX à partir du meilleur point de la
   grille.
2. **Probabilité de succès** : P = fraction des tirages du posterior pour
   lesquels cerveau(theta_s; d, h) <= 0.2.
3. **Version robuste** : minimiser la distance sous la contrainte
   P(succès) >= 0.9. Celle-ci équivaut à : le quantile à 90 % de cerveau(theta_s;
   d, h) sur les tirages est <= 0.2. Le quantile est continu et presque partout
   dérivable, donc utilisable par SLSQP.
4. **Séparation optimisation / évaluation** : on optimise sur 1000 tirages et on
   évalue P sur 3000 autres. Sinon, l'optimiseur pourrait exploiter le bruit de
   l'échantillon (surapprentissage) et la probabilité annoncée serait trop
   optimiste.
5. **Solution exacte expm** : le système est linéaire, donc
   y(24 h) = expm(24 A) y0. C'est environ 40 fois plus rapide que diffrax
   (160 microsecondes contre 6 ms par évaluation). La carte demande
   41 x 41 x 1000 = 1.7 million d'évaluations. Un test vérifie l'accord avec
   diffrax (rtol 1e-6).

### 5.5 Résultats

| Candidat | d (nm) | h | Distance | Cerveau à la moyenne | P(succès) | IC 90 % du cerveau | Vérité |
|---|---|---|---|---|---|---|---|
| Particule d'origine | 5.00 | 1.00 | 0 | 0.294 | 0.000 | [0.256, 0.351] | 0.286 |
| Point estimé (optimal) | 6.87 | 1.57 | 1.14 | 0.200 | **0.441** | [0.164, 0.257] | 0.192 |
| Point estimé, d seul | 7.86 | 1.00 | 1.43 | 0.200 | 0.417 | [0.174, 0.241] | 0.197 |
| h maximal (h = 8), d inchangé | 5.00 | 8.00 | 3.00 | 0.214 | 0.386 | [0.132, 0.332] | 0.186 |
| Point estimé, marge x0.8 | 7.81 | 1.75 | 1.62 | 0.160 | 0.915 | [0.129, 0.209] | 0.153 |
| **Robuste, P >= 0.9** | 7.83 | 1.59 | 1.57 | 0.165 | **0.891** | [0.135, 0.213] | 0.159 |

« Vérité » est la quantité calculée avec les vrais paramètres. On ne la connaît
pas en pratique ; on peut l'afficher ici seulement parce que les données sont
synthétiques.

### 5.6 Interprétation

**1. Le point estimé dit « réglé », mais P(succès) = 0.44.** La solution au
point estimé se place exactement sur le seuil pour theta moyen. Par
construction, elle échoue donc pour environ la moitié des valeurs plausibles des
paramètres. Ce n'est pas une réponse fiable. (Ici, elle aurait fonctionné avec
les vrais paramètres, 0.192 < 0.2, mais c'est de la chance : on ne peut pas le
savoir à l'avance.)

**Pourquoi P = 0.44 et pas 0.50 ?** On a mis la moyenne des paramètres dans un
modèle non linéaire. La sortie obtenue n'est ni la moyenne ni la médiane des
sorties (inégalité de Jensen). Ici, cerveau(theta_moyen) est un peu sous la
médiane de la distribution prédictive, d'où P < 0.5.

**2. Agir seulement sur h : un levier puissant mais incertain.** À la moyenne du
posterior, même h = 8 ne suffit pas (0.214 > 0.2), et P = 0.39. Surtout, l'IC
à 90 % est très large : [0.132, 0.332]. L'effet de h passe par m_brain, le
paramètre le moins bien identifié (phase 4, contraction 0.40). Si m_brain est
petit, coater plus vite protège beaucoup le cerveau ; s'il est proche de 1,
presque pas. Les données ne permettent pas de trancher, donc on ne peut pas
compter sur h.

**3. La solution robuste.** Distance 1.57 (contre 1.14) : il faut aller plus loin,
et **surtout dans la direction de d** (u passe de 0.94 à 1.42, alors que v
reste presque le même : 0.65 contre 0.67). d agit via CL_renal et k_in_brain,
deux paramètres bien déterminés. **L'incertitude ne change pas seulement la
distance, elle change aussi la direction de la solution.**

**4. Robuste contre « marge de sécurité ».** Viser 0.8 x seuil au point estimé
donne ici P = 0.915, pour une distance de 1.62. La solution robuste fait un peu
mieux (1.57, P = 0.89). Mais la marge de 20 % a été choisie au hasard. Avec 10 %,
P aurait été plus faible ; avec 40 %, on aurait payé un changement inutilement
grand. L'approche robuste règle directement ce qui compte (la probabilité de
succès) et choisit la direction en fonction de l'incertitude. La marge ne le fait
pas.

**5. Vérification hors échantillon.** P = 0.900 sur les 1000 tirages
d'optimisation, contre 0.891 sur les 3000 tirages réservés. L'écart-type
d'échantillonnage d'une proportion de 0.9 sur 3000 tirages vaut environ 0.0055,
donc l'écart (0.009) est d'environ 1.6 écart-type. C'est l'effet attendu
de l'optimisation sur un échantillon fini, qui exploite un peu son bruit. Pour
une garantie de 0.9 hors échantillon, il faudrait viser un peu plus haut
(par exemple 0.92) ou optimiser sur davantage de tirages.

**6. Retour sur ma prévision de la phase 4.** J'avais prévu que d serait bien
plus efficace que h. C'est vrai en termes de fiabilité, mais pas en termes
d'effet au point estimé : h a un effet moyen important, simplement très
incertain. Les indices de Sobol (phase 4) faisaient varier k_coat d'un facteur
environ 2.7 (prior), alors qu'ici h varie jusqu'à un facteur 8, avec theta fixé
au posterior. Les deux analyses répondent à des questions différentes.

### 5.7 Lecture de la figure `figures/phase5_success_map.png`

- Couleur : P(succès) en chaque point (d, h).
- Tiret orange : là où le point estimé atteint juste le seuil. Il suit presque
  la courbe P = 0.5 (ligne grise).
- Ligne noire : P = 0.9, frontière de la région robuste.
- Pointillés : cercles de distance égale à celle des deux solutions.
- Carré : point estimé. Étoile : robuste. Triangle : h = 8 seul.
- La frontière P = 0.9 est presque verticale pour h grand : au-delà d'un
  certain point, augmenter h n'apporte plus de garantie. C'est d qui fait le
  travail.

### 5.8 Questions probables en entretien

**Q : Qu'est-ce qu'un contrefactuel ici ?**
R : C'est la réponse à « qu'aurait-on observé si la particule avait eu un autre
diamètre ou une autre chimie de surface ? ». On répond en intervenant sur les
paramètres du modèle mécaniste calibré, à travers des lois explicites
propriétés vers taux. On ne s'appuie pas sur des corrélations dans les
données. C'est la logique de l'opérateur do() de Pearl, appliquée à un modèle
causal structurel qui est ici le système d'EDO.

**Q : Pourquoi ne pas faire confiance au point estimé ?**
R : Parce que la solution est placée exactement sur le seuil pour un seul jeu
de paramètres. Avec l'incertitude réelle, elle échoue dans 56 % des cas. Si on
fabrique cette particule et qu'on la teste, on a moins d'une chance sur deux
qu'elle satisfasse la contrainte.

**Q : Votre P(succès) est-il une vraie probabilité ?**
R : C'est une probabilité a posteriori, conditionnelle au modèle, aux priors et
aux lois propriétés vers taux. Elle ne tient pas compte de l'erreur de
structure du modèle, ni de l'incertitude sur les constantes des lois (6 nm,
4 nm), que je suppose connues. En pratique, j'ajouterais des priors sur ces
constantes, ce qui élargirait les distributions et rendrait la solution robuste
plus coûteuse.

**Q : Pourquoi la contrainte sur le quantile ?**
R : P(cerveau <= seuil) >= 0.9 équivaut exactement à « quantile à 90 % de
cerveau <= seuil ». Le quantile est une fonction continue de (d, h), alors que
la proportion est une fonction en escalier, sans gradient utile. C'est la
formulation classique des contraintes probabilistes (chance constraints).

**Q : Pourquoi une recherche sur grille avant SLSQP ?**
R : Le problème n'est pas convexe (effet non monotone de d) : un optimiseur local
peut converger vers un mauvais minimum. La grille donne un bon point de départ
global, et SLSQP affine la solution. Pour le point estimé, le meilleur point de
la grille est (u, v) = (1.10, 0.45), à la distance 1.19 ; SLSQP trouve
(0.94, 0.65), à la distance 1.14, un peu plus loin le long de la même frontière.

**Q : Pourquoi exp(A t) et pas diffrax ici ?**
R : Pour la vitesse : 1.7 million d'évaluations, environ 40 fois plus rapides,
et un résultat exact pour un système linéaire. C'est vérifié par un test. Pour
un modèle non linéaire, je construirais un émulateur, par exemple un processus
gaussien entraîné sur quelques centaines de simulations, en propageant son
incertitude dans P(succès).

**Q : Que changeriez-vous avec de vraies données ?**
R : Trois choses. (1) Calibrer les lois propriétés vers taux sur plusieurs
particules de tailles et de chimies différentes, au lieu de les supposer.
(2) Faire l'expérience du bras pré-coaté (phase 4), qui fixerait m_brain et
rendrait h utilisable comme levier. (3) Tester la particule robuste in vitro
(tissue-on-chip), puis mettre à jour le posterior avec ce résultat : c'est une
boucle d'apprentissage actif.

---

## Phase 6 : packaging, reproductibilité et présentation

### 6.1 Ce qui rend le projet reproductible

| Élément | Rôle |
|---|---|
| `pyproject.toml` en *src layout* | Le code est un vrai paquet installable (`pip install -e .`). Le dossier `src/` évite d'importer par erreur le code local au lieu du paquet installé, ce qui garantit que les tests vérifient ce qui est réellement installé. |
| `pip install -e .` (mode éditable) | Le paquet pointe vers les sources : une modification du code est visible sans réinstaller. |
| `requirements-lock.txt` | Versions exactes utilisées (jax 0.10.2, diffrax 0.7.2, numpyro 0.22.0, etc.). `pyproject.toml` donne des versions minimales (souples), le fichier lock les versions exactes (reproductibles). |
| Graines fixées | Données (graine 20261102), NUTS (graine 0), tirages, Sobol, séparation des tirages : relancer donne les mêmes chiffres. |
| `scripts/run_all.py` | Une seule commande refait toutes les figures et tous les chiffres. |
| Un fichier `results/*.txt` par phase | Chaque chiffre du README vient d'un fichier écrit par un script : rien n'est recopié à la main. |
| 32 tests (`pytest`) | Accord entre solveurs, conservation de la masse, estimateurs de Sobol sur des cas analytiques, lois des propriétés, accord expm / diffrax. |
| `notebooks/demo.ipynb` | Une visite guidée exécutable de bout en bout, qui réutilise le posterior déjà calculé. |

### 6.2 Résumé du projet en 2 minutes (pour l'entretien)

« Une nanoparticule injectée change d'identité : des protéines du plasma
forment une couronne, et la particule coatée ne va pas aux mêmes endroits que
la particule nue. J'ai construit un modèle à compartiments (sang et cinq
organes) où chaque particule existe dans deux états, avec des EDO résolues en
JAX/diffrax et vérifiées contre une solution exacte. J'ai simulé une étude de
biodistribution réaliste, avec peu de points et du bruit, puis calibré les 12
paramètres avec NumPyro et NUTS : les diagnostics sont propres, et toutes les
valeurs vraies sont dans les intervalles de crédibilité à 90 %. L'analyse
d'identifiabilité montre que l'effet de la couronne sur le cerveau est mal
déterminé, parce qu'on n'observe que des totaux. Avec l'information de Fisher,
je montre qu'un second groupe recevant des particules pré-coatées le fixerait.
Enfin, je réponds à une question de conception : quel plus petit changement de
taille et de chimie de surface réduit l'accumulation cérébrale ? La solution
au point estimé ne marche que dans 44 % des cas plausibles. La solution robuste
atteint 89 % et change surtout la taille, car c'est le levier dont l'effet est
bien connu. »

### 6.3 Lien avec le projet SYNERGIE

| Demande du poste | Ce que Metamorph montre |
|---|---|
| Modèles par EDO / systèmes dynamiques | Modèle à 14 états, analyse des valeurs propres (raideur), trois solveurs concordants, conservation de la masse |
| Inférence bayésienne dans un langage probabiliste | NumPyro, NUTS, checks prédictifs a priori et a posteriori, R-hat, ESS, divergences, diagnostic de la géométrie (matrice de masse dense) |
| Python scientifique solide | Paquet installable, typé, documenté, testé, reproductible |
| Optimisation | SLSQP sous contraintes, contraintes probabilistes (chance constraints), recherche globale puis locale |
| Raisonnement contrefactuel | Intervention sur des propriétés via un modèle mécaniste, avec propagation de l'incertitude |
| Modèles de substitution (surrogates) | Solution exacte expm comme « surrogate » exact ; un émulateur par processus gaussien est proposé pour le cas non linéaire |
| Transport dans les tissus humains | Version miniature : compartiments, changement d'état, barrières (cerveau), clairance |

### 6.4 Questions probables en entretien

**Q : Qu'est-ce que vous feriez différemment avec plus de temps ?**
R : Trois choses. (1) Tester le modèle sur des données générées par un modèle
plus riche (par exemple avec captation saturable), pour voir comment
l'inférence réagit à une erreur de structure : ici le modèle ajusté est le
vrai, ce qui est optimiste. (2) Mettre des priors sur les constantes des lois
propriétés vers taux. (3) Faire un vrai plan d'expérience bayésien, avec le
gain d'information attendu comme critère.

**Q : Quelle est la principale faiblesse de ce travail ?**
R : Les données sont synthétiques, générées par la même structure de modèle.
La méthode est validée (on retrouve les vrais paramètres), mais pas le modèle
lui-même. C'est exactement ce que le doctorat apporterait : des données réelles
(tissue-on-chip) et une structure physiologique (PBPK).

**Q : Comment passeriez-vous à un modèle PBPK avec de vraies données ?**
R : Les compartiments deviendraient des volumes d'organes reliés par des débits
sanguins. Les paramètres deviendraient des perméabilités et des coefficients de
partage, avec des priors issus de la physiologie. Le modèle grossit : je
garderais NUTS si possible, avec une attention particulière à
l'identifiabilité, et j'utiliserais un émulateur pour l'optimisation. Un
niveau hiérarchique permettrait de tenir compte de la variabilité entre
individus ou entre puces.

**Q : Pourquoi JAX ?**
R : Pour la différentiation automatique à travers le solveur d'EDO (nécessaire
pour NUTS et pour l'information de Fisher), la vectorisation (vmap) pour
simuler des milliers de jeux de paramètres d'un coup, et la compilation (jit).
Un même code sert à la simulation, à l'inférence et à l'optimisation.

**Q : Combien de temps tout cela prend-il à calculer ?**
R : 16.6 minutes sur un ordinateur portable pour tout refaire, dont 9.4 pour
NUTS, 4.3 pour l'identifiabilité et la sensibilité, et 2.3 pour le
contrefactuel (`results/run_all_timings.txt`, écrit par `scripts/run_all.py`).

**Q : Le code est-il fiable ?**
R : 32 tests automatiques : deux implémentations indépendantes du modèle, une
solution exacte, la conservation de la masse pour des paramètres aléatoires,
les estimateurs de Sobol vérifiés sur des fonctions dont les indices sont
connus, et l'accord entre la solution exacte et diffrax. De plus, tout le
pipeline a été relancé de bout en bout avec `run_all.py` : les figures, les
données et tous les chiffres scientifiques sont identiques à ceux commités
(les tirages du posterior sont identiques au bit près). Seuls les temps de
calcul changent.

**Q : Le notebook donne une solution robuste légèrement différente du script
de la phase 5. Pourquoi ?**
R : Le notebook part d'une grille plus grossière (21 x 21 au lieu de 41 x 41)
pour aller vite. Il trouve (d, h) = (7.96 nm, 1.45) au lieu de (7.83 nm, 1.59),
à la même distance (1.57), avec P = 0.892 contre 0.891. La frontière robuste est
presque plate autour de l'optimum : plusieurs points voisins sont
équivalents. La conclusion ne change pas : la solution robuste va surtout le
long de d. C'est aussi une leçon : quand l'optimum est plat, il faut rapporter
la région de solutions équivalentes, pas un point avec une fausse précision.
