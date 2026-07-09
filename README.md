# Construire un LLM de zéro

De l'addition de deux vecteurs jusqu'à un agent qui raisonne, ce livre construit un modèle de langage entièrement à la main, en français. Chaque chapitre se lit sur [llmdezero.com](https://llmdezero.com) (avec ses **visualisations interactives**) et se code dans le notebook correspondant de ce dépôt.

## Le pacte

Les notebooks de ce dépôt sont gratuits et le resteront. Tu peux les ouvrir, les exécuter et apprendre avec, sans rien payer. Le livre en ligne sur [llmdezero.com](https://llmdezero.com) les accompagne : il porte le texte, les explications et les **visualisations interactives** qui donnent leur sens à ces exercices. Chaque notion délicate y a la sienne : tu déplaces un curseur, tu changes un nombre, et tu vois l'effet en direct (l'attention qui se concentre, la perte qui descend, les probabilités qui se déforment). Les notebooks sont la pratique, le livre est le fil.

## Comment travailler

Le notebook de chaque chapitre contient tout le code de la leçon, complet et exécutable : lis, exécute, modifie pour voir. Il se termine par une section **Exercices**, graduée du plus simple au plus costaud : les marqueurs `TODO(toi)` y signalent les endroits où c'est à toi de coder, et des cellules `assert` valident chaque réponse. Une fois bloqué, ou pour vérifier, ouvre la version `solution` du même chapitre dans le dossier `solutions/` : elle contient les corrigés des exercices.

Deux façons d'exécuter un notebook :

- **Google Colab** (recommandé, rien à installer) : clique sur le lien « Ouvrir dans Colab » de la ligne du chapitre.
- **En local** :
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  jupyter lab
  ```

## Les 22 chapitres

| Ch. | Titre | Notebook | Solution |
|-----|-------|----------|----------|
| 0 | Le Python dont tu as besoin | [Ouvrir dans Colab](https://colab.research.google.com/github/kevindegila/construire-un-llm-de-zero/blob/main/avant_de_commencer/chapitre_00_python.ipynb) | [solution](solutions/avant_de_commencer/chapitre_00_python_solution.ipynb) |
| 1 | Faire parler une machine | [Ouvrir dans Colab](https://colab.research.google.com/github/kevindegila/construire-un-llm-de-zero/blob/main/partie_1_etincelle/chapitre_01_faire_parler_une_machine.ipynb) | [solution](solutions/partie_1_etincelle/chapitre_01_faire_parler_une_machine_solution.ipynb) |
| 2 | Les nombres qui apprennent | [Ouvrir dans Colab](https://colab.research.google.com/github/kevindegila/construire-un-llm-de-zero/blob/main/partie_1_etincelle/chapitre_02_les_nombres_qui_apprennent.ipynb) | [solution](solutions/partie_1_etincelle/chapitre_02_les_nombres_qui_apprennent_solution.ipynb) |
| 3 | Comment une machine apprend | [Ouvrir dans Colab](https://colab.research.google.com/github/kevindegila/construire-un-llm-de-zero/blob/main/partie_1_etincelle/chapitre_03_comment_une_machine_apprend.ipynb) | [solution](solutions/partie_1_etincelle/chapitre_03_comment_une_machine_apprend_solution.ipynb) |
| 4 | Construire un moteur d'autograd | [Ouvrir dans Colab](https://colab.research.google.com/github/kevindegila/construire-un-llm-de-zero/blob/main/partie_1_etincelle/chapitre_04_moteur_autograd.ipynb) | [solution](solutions/partie_1_etincelle/chapitre_04_moteur_autograd_solution.ipynb) |
| 5 | Un peu de hasard | [Ouvrir dans Colab](https://colab.research.google.com/github/kevindegila/construire-un-llm-de-zero/blob/main/partie_1_etincelle/chapitre_05_un_peu_de_hasard.ipynb) | [solution](solutions/partie_1_etincelle/chapitre_05_un_peu_de_hasard_solution.ipynb) |
| 6 | Le neurone et le réseau | [Ouvrir dans Colab](https://colab.research.google.com/github/kevindegila/construire-un-llm-de-zero/blob/main/partie_2_construire_le_cerveau/chapitre_06_le_neurone_et_le_reseau.ipynb) | [solution](solutions/partie_2_construire_le_cerveau/chapitre_06_le_neurone_et_le_reseau_solution.ipynb) |
| 7 | Découper le langage | [Ouvrir dans Colab](https://colab.research.google.com/github/kevindegila/construire-un-llm-de-zero/blob/main/partie_2_construire_le_cerveau/chapitre_07_decouper_le_langage.ipynb) | [solution](solutions/partie_2_construire_le_cerveau/chapitre_07_decouper_le_langage_solution.ipynb) |
| 8 | Donner du sens aux mots | [Ouvrir dans Colab](https://colab.research.google.com/github/kevindegila/construire-un-llm-de-zero/blob/main/partie_2_construire_le_cerveau/chapitre_08_donner_du_sens_aux_mots.ipynb) | [solution](solutions/partie_2_construire_le_cerveau/chapitre_08_donner_du_sens_aux_mots_solution.ipynb) |
| 9 | L'attention | [Ouvrir dans Colab](https://colab.research.google.com/github/kevindegila/construire-un-llm-de-zero/blob/main/partie_2_construire_le_cerveau/chapitre_09_attention.ipynb) | [solution](solutions/partie_2_construire_le_cerveau/chapitre_09_attention_solution.ipynb) |
| 10 | Le Transformer | [Ouvrir dans Colab](https://colab.research.google.com/github/kevindegila/construire-un-llm-de-zero/blob/main/partie_2_construire_le_cerveau/chapitre_10_le_transformer.ipynb) | [solution](solutions/partie_2_construire_le_cerveau/chapitre_10_le_transformer_solution.ipynb) |
| 11 | Bien s'entraîner | [Ouvrir dans Colab](https://colab.research.google.com/github/kevindegila/construire-un-llm-de-zero/blob/main/partie_3_sentrainer_comme_un_labo/chapitre_11_bien_sentrainer.ipynb) | [solution](solutions/partie_3_sentrainer_comme_un_labo/chapitre_11_bien_sentrainer_solution.ipynb) |
| 12 | Survivre à Colab | [Ouvrir dans Colab](https://colab.research.google.com/github/kevindegila/construire-un-llm-de-zero/blob/main/partie_3_sentrainer_comme_un_labo/chapitre_12_survivre_a_colab.ipynb) | [solution](solutions/partie_3_sentrainer_comme_un_labo/chapitre_12_survivre_a_colab_solution.ipynb) |
| 13 | Les données | [Ouvrir dans Colab](https://colab.research.google.com/github/kevindegila/construire-un-llm-de-zero/blob/main/partie_3_sentrainer_comme_un_labo/chapitre_13_les_donnees.ipynb) | [solution](solutions/partie_3_sentrainer_comme_un_labo/chapitre_13_les_donnees_solution.ipynb) |
| 14 | Compter ce qui coûte | [Ouvrir dans Colab](https://colab.research.google.com/github/kevindegila/construire-un-llm-de-zero/blob/main/partie_3_sentrainer_comme_un_labo/chapitre_14_compter_ce_qui_coute.ipynb) | [solution](solutions/partie_3_sentrainer_comme_un_labo/chapitre_14_compter_ce_qui_coute_solution.ipynb) |
| 15 | Passer à l'échelle | [Ouvrir dans Colab](https://colab.research.google.com/github/kevindegila/construire-un-llm-de-zero/blob/main/partie_3_sentrainer_comme_un_labo/chapitre_15_passer_a_lechelle.ipynb) | [solution](solutions/partie_3_sentrainer_comme_un_labo/chapitre_15_passer_a_lechelle_solution.ipynb) |
| 16 | Évaluer sérieusement | [Ouvrir dans Colab](https://colab.research.google.com/github/kevindegila/construire-un-llm-de-zero/blob/main/partie_3_sentrainer_comme_un_labo/chapitre_16_evaluer_serieusement.ipynb) | [solution](solutions/partie_3_sentrainer_comme_un_labo/chapitre_16_evaluer_serieusement_solution.ipynb) |
| 17 | Apprendre à obéir | [Ouvrir dans Colab](https://colab.research.google.com/github/kevindegila/construire-un-llm-de-zero/blob/main/partie_4_du_modele_a_lesprit/chapitre_17_apprendre_a_obeir.ipynb) | [solution](solutions/partie_4_du_modele_a_lesprit/chapitre_17_apprendre_a_obeir_solution.ipynb) |
| 18 | Apprendre des préférences | [Ouvrir dans Colab](https://colab.research.google.com/github/kevindegila/construire-un-llm-de-zero/blob/main/partie_4_du_modele_a_lesprit/chapitre_18_apprendre_des_preferences.ipynb) | [solution](solutions/partie_4_du_modele_a_lesprit/chapitre_18_apprendre_des_preferences_solution.ipynb) |
| 19 | La frontière | [Ouvrir dans Colab](https://colab.research.google.com/github/kevindegila/construire-un-llm-de-zero/blob/main/partie_4_du_modele_a_lesprit/chapitre_19_la_frontiere.ipynb) | [solution](solutions/partie_4_du_modele_a_lesprit/chapitre_19_la_frontiere_solution.ipynb) |
| 20 | Faire tourner un modèle | [Ouvrir dans Colab](https://colab.research.google.com/github/kevindegila/construire-un-llm-de-zero/blob/main/partie_4_du_modele_a_lesprit/chapitre_20_faire_tourner_un_modele.ipynb) | [solution](solutions/partie_4_du_modele_a_lesprit/chapitre_20_faire_tourner_un_modele_solution.ipynb) |
| 21 | Raisonner et agir | [Ouvrir dans Colab](https://colab.research.google.com/github/kevindegila/construire-un-llm-de-zero/blob/main/partie_4_du_modele_a_lesprit/chapitre_21_raisonner_et_agir.ipynb) | [solution](solutions/partie_4_du_modele_a_lesprit/chapitre_21_raisonner_et_agir_solution.ipynb) |

Les chapitres sont regroupés en quatre parties : partie 1, L'étincelle (ch. 1 à 5) ; partie 2, Construire le cerveau (ch. 6 à 10) ; partie 3, S'entraîner comme un labo (ch. 11 à 16) ; partie 4, Du modèle à l'esprit (ch. 17 à 21). Le chapitre 0 (Le Python dont tu as besoin) est un sas de mise à niveau facultatif.

## Prérequis

Installe les dépendances avec `pip install -r requirements.txt`. Le cœur tient en trois bibliothèques : `torch`, `numpy` et `matplotlib`, plus `jupyterlab` pour exécuter les notebooks en local. Quelques chapitres de la partie 4 peuvent charger des bibliothèques plus lourdes (`transformers`, `datasets`, `bitsandbytes`) : elles sont listées en commentaire dans `requirements.txt`, à décommenter seulement si tu en as besoin.

## Tout tourne hors ligne

Tous les notebooks s'exécutent sans GPU, sur une machine ordinaire ou dans un Colab gratuit. La seule exception est la section cloud du chapitre 15 (Passer à l'échelle), qui montre l'entraînement multi-GPU sur un nœud loué : cette partie demande du matériel réel, et le reste du chapitre reste faisable hors ligne.

## Citation

Si ce livre ou son code te sont utiles pour tes travaux, tu peux le citer.

Citation (style Chicago) :

> Degila, Kévin. *Construire un LLM de zéro*. 2026. https://llmdezero.com.

Entrée BibTeX :

```bibtex
@book{degila2026construire,
  author  = {Kévin Degila},
  title   = {Construire un LLM de zéro},
  year    = {2026},
  url     = {https://llmdezero.com},
  note    = {Code et notebooks : https://github.com/kevindegila/construire-un-llm-de-zero}
}
```

## Licence

Le code de ce dépôt (notebooks et solutions) est distribué sous **licence Apache 2.0** (voir le fichier [`LICENSE`](LICENSE)). Tu es libre de l'utiliser, le modifier et le redistribuer, y compris à des fins commerciales, tant que tu conserves la notice de licence et que tu signales tes modifications. Copyright 2026 Kévin Degila.
