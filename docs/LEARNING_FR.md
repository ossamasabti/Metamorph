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
