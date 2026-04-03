# ftry

CLI minimale nommee `ftry`.

## Commandes disponibles

Les commandes sont pour l'instant des mocks, sauf `line`.

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
