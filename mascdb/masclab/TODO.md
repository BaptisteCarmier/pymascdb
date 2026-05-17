- Les calculs sont pas bons pour les paramètres d'elipses, etc ... --> A vérifier quand on aura une version stable en matlab (et que j'arrive à run)

- faire en sorte que tout soit compatible avec masc_mat_file_to_dict() de pymascdb, réussir à lancer process_all()

- faire en sorte que les triplets fonctionnent et qu'ils renvoient la bonne chose (prendre un code matlab en comparaison) -- prendre exemple sur un code matlab

- faire la classification et les quicklooks associés -- prendre exemple sur un code matlab

- faire le blowing snow

- changer le path des images processes etc ... pour que tout soit en relatif. Idéalement, y'a le path de la campagne, puis dans chaque 

- Changer uploaddirs.py et runner_joblib.py si on veut looper sur les campagnes d'abord. Dépend de si on veut run le code campagnes par campagnes ou toutes les campagnes d'un coup. A poser la question ! 

- Changer le saving des données pour que ça soit en .joblib tout du long, et à la fin faire une conversion en .mat pour la lecture/compatibilité
