# Phase 1 : proposition de paramètres (version 2)

Statut : proposition, en attente de validation. Aucun code n'a encore été écrit.

## 1. Conventions

- Unités : quantités en pourcentage de la dose injectée (%ID), temps en heures.
- Horizon : 0 à 24 h. Le critère sur le cerveau en phase 5 est évalué à t_end = 24 h.
- Condition initiale : injection IV de particules nues, B_b(0) = 100, tous les autres états à 0.
- États (14) :
  - sang : B_b, B_c ;
  - tissus : T_{i,b} et T_{i,c} pour i dans {poumon, foie, rate, rein, cerveau} (10 états) ;
  - compartiments cumulatifs d'élimination : A_renal et A_hepatic.
- Avec A_renal et A_hepatic, la masse totale vaut exactement 100 à tout instant. Le test de
  conservation s'écrit donc : somme des 14 états = 100 (à la tolérance du solveur près).

## 2. Équations

Pour s dans {b, c} et i dans les 5 tissus :

```
dB_b/dt      = -k_coat*B_b - sum_i k_in[i,b]*B_b + sum_i k_out[i]*T_i,b - CL_renal*B_b
dB_c/dt      =  k_coat*B_b - sum_i k_in[i,c]*B_c + sum_i k_out[i]*T_i,c
dT_i,s/dt    =  k_in[i,s]*B_s - k_out[i]*T_i,s - CL_liver*T_i,s * [i = foie]
dA_renal/dt  =  CL_renal*B_b
dA_hepatic/dt = CL_liver*(T_foie,b + T_foie,c)
```

Paramétrage de l'état coaté : k_in[i,c] = k_in[i,b] * m_i, avec :

- m_foie = m_rate = m_RES (paramètre inféré) ;
- m_cerveau = m_brain (paramètre inféré) ;
- m_poumon = m_rein = 1 (fixés).

Paramétrage du retour vers le sang :

- k_out[poumon] = k_out[rein] = k_out_fast ;
- k_out[foie] = k_out[rate] = k_out[cerveau] = k_out_slow.

## 3. Les 12 paramètres inférés

Unités : h^-1, sauf les multiplicateurs (sans dimension).

| #  | Paramètre      | Valeur vraie | Justification |
|----|----------------|--------------|---------------|
| 1  | k_coat         | 1.5   | Changement d'état nu vers coaté. Demi-vie ln2/1.5 = 0.46 h, environ 28 min (voir section 5). |
| 2  | CL_renal       | 0.8   | Filtration rénale des particules nues dans le sang. Levier principal du diamètre d en phase 5. |
| 3  | CL_liver       | 0.05  | Élimination hépatobiliaire lente, demi-vie environ 14 h. |
| 4  | k_in_lung      | 0.3   | Organe très perfusé, captation rapide. |
| 5  | k_in_liver     | 0.4   | Captation par les cellules de Kupffer. |
| 6  | k_in_spleen    | 0.1   | Captation par les macrophages, organe plus petit que le foie. |
| 7  | k_in_kidney    | 0.2   | Captation tissulaire rénale, distincte de la filtration. |
| 8  | k_in_brain     | 0.01  | Barrière hémato-encéphalique : captation très faible. |
| 9  | m_RES          | 3.0   | Multiplicateur coaté partagé foie/rate : l'opsonisation favorise la reconnaissance par les macrophages. |
| 10 | m_brain        | 0.3   | La couronne protéique réduit le passage de la barrière hémato-encéphalique. |
| 11 | k_out_fast     | 0.5   | Retour vers le sang depuis les tissus d'échange (poumon, rein). |
| 12 | k_out_slow     | 0.03  | Retour lent depuis les tissus qui séquestrent (foie, rate) et efflux lent du cerveau. |

Paramètre de nuisance ajouté en phase 3 : sigma, écart-type du bruit log-normal multiplicatif.

## 4. Vérification rapide (simulation SciPy hors dépôt)

Quantités en %ID. La masse totale vaut 100.0000 à chaque instant.

| t (h) | Sang  | Poumon | Foie  | Rate  | Rein | Cerveau | A_renal | A_hepatic |
|-------|-------|--------|-------|-------|------|---------|---------|-----------|
| 0.25  | 63.70 | 5.65   | 10.39 | 2.61  | 3.76 | 0.18    | 13.64   | 0.06      |
| 1     | 18.60 | 10.35  | 30.90 | 7.95  | 6.90 | 0.34    | 24.05   | 0.90      |
| 4     | 3.27  | 4.83   | 41.66 | 12.03 | 3.22 | 0.40    | 27.79   | 6.79      |
| 8     | 1.61  | 1.70   | 38.29 | 12.91 | 1.14 | 0.39    | 29.09   | 14.87     |
| 24    | 0.66  | 0.42   | 19.47 | 11.26 | 0.28 | 0.29    | 30.37   | 37.26     |

Ces ordres de grandeur sont plausibles : le foie et la rate dominent, l'élimination rénale
est importante pour une petite particule, et le cerveau reste sous 1 %ID.

Ces chiffres viennent d'un script exploratoire. Ils ne seront pas repris dans le README :
les chiffres définitifs seront produits par `scripts/` en phase 1.

## 5. Points anticipés (suite à la relecture)

### 5.1 k_coat plus lent que la réalité

Dans la littérature, la couronne protéique se forme en quelques secondes à quelques minutes.
Ici, k_coat ne représente pas l'adsorption initiale (couronne molle) mais un changement d'état
fonctionnel effectif : la maturation de la couronne dure, qui modifie la reconnaissance par
les macrophages et le passage des barrières. La valeur est choisie volontairement pour que
les deux états restent visibles sur l'échelle de temps des mesures. Cette hypothèse sera
écrite dans la section "Model assumptions and limitations" du README.

### 5.2 Identifiabilité entre k_coat et CL_renal

Les deux paramètres vident B_b. Avec les seules quantités tissulaires totales, ils seraient
mal séparés. Décision pour la phase 2 : on observe en plus

- la quantité totale dans le sang, B_b + B_c (prélèvements plasmatiques) ;
- l'élimination urinaire cumulée, A_renal (collecte d'urine, standard dans les études
  de biodistribution).

A_renal contraint directement CL_renal (intégrale de CL_renal*B_b), ce qui libère k_coat.

### 5.3 Non-identifiabilité attendue entre k_out_slow et CL_liver

Le contenu du foie dépend surtout de la somme k_out_slow + CL_liver. Les deux ne se séparent
que via la rate et le cerveau, qui partagent k_out_slow. Si A_hepatic (élimination fécale)
n'est pas observée, on s'attend à une corrélation négative dans le posterior. C'est un cas
illustratif prévu pour la phase 4. On pourra montrer qu'observer A_hepatic la lève.

## 6. Décisions restant à valider

1. Les 12 paramètres et leurs valeurs vraies (section 3).
2. Les observables de la phase 2 : tissus totaux, sang total, A_renal cumulé (section 5.2).
   A_hepatic reste non observé volontairement (section 5.3).
3. L'horizon de 24 h et t_end = 24 h pour la phase 5.
