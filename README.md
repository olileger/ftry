# ftry

CLI minimale nommee `ftry`.

## Commandes disponibles

Les commandes sont pour l'instant des mocks, sauf `line` et `pop`.

- `ftry build`
- `ftry break`
- `ftry pop`
- `ftry land`
- `ftry line`

Exemple:

```powershell
ftry build
```

Sortie:

```text
build
```

La commande `ftry line` charge son rendu depuis `src\ftry\line.txt`. Pour changer le visuel, il suffit donc de modifier ce fichier.

La commande `ftry pop` charge soit un agent (`-a`), soit une team d'agents (`-t`) depuis un fichier YAML, envoie le prompt passe avec `-p`, puis affiche la reponse du modele.

Exemple:

```
ftry pop -a .\samples\poete.yaml -p "Ecris un poeme sur la pluie"
```

Exemple team:

```powershell
ftry pop -t .\samples\team.yaml -p "Construis un meilleur prompt pour resumer ce texte"
```

## Installation locale

Prerequis:

- Python 3.10 ou plus recent
- `pip`

Depuis la racine du projet, installer la CLI en mode local editable:

```powershell
python -m pip install -e .
```

Ensuite, la commande est disponible dans le terminal:

```powershell
ftry break
```
