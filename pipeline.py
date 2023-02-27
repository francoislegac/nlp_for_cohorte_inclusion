import spacy
import pandas as pd


def pipeline(txt):
    nlp = spacy.blank("fr")
    regex = dict(epilepsie=[r"[eé]pileptiques?"])
    # terms = dict(epilepsie="épileptique")
    # Sentencizer component, needed for negation detection
    nlp.add_pipe("eds.sentences")
    nlp.add_pipe("eds.normalizer")
    nlp.add_pipe(
        "eds.matcher",
        config=dict(
            regex=regex,
            # terms=terms,
            attr="LOWER",
            ignore_excluded=True,
        ),
    )
    nlp.add_pipe("eds.negation")
    nlp.add_pipe("eds.family")
    nlp.add_pipe("eds.hypothesis")

    entities = []
    doc = nlp(txt)
    for ent in doc.ents:
        d = dict(
            lexical_variant=ent.text,
            start_char=ent.start_char,
            end_char=ent.end_char,
            label=ent.label_,
            sent=ent.sent,
            negation=ent._.negation,
            hypothesis=ent._.hypothesis,
            family=ent._.family,
        )
        entities.append(d)

    df = pd.DataFrame.from_records(entities)
    label = 0
    # check si on a au moins une mention "épileptique" et si neg,
    if True in list(~df.all(axis=1)):
        label = 1
    return (df, label)


def test_pipeline():
    # text vide
    # cas entité avec / sans accent, pluriel ?
    # cas majuscules
    # cas sur 2 lignes
    pass


if __name__ == "__main__":
    txt = """Bonjour\nJ\'ai 48 ans et je suis devenu épileptique 
    à l\'âge de 24 ans.\nJ\'ai testé presque tous les médicaments (lamictale, Phénobarbital, Dépakine ...), 
    les doses, les mélanges, etc. mais ma maladie reste pharmaco-résistante .\nElle touche le globe temporal gauche.\nAprès deux "grand mal" qui ont brisé mes bras, 
    je ne fais plus que de "petites crises" : perte de conscience, bras qui s\'agite, ...
    \nJe suis prof et ça m\'arrive parfxois en classe. 
    Mais je préviens toujours mes élèves en début d\'année.
    \nUn an d\'examens type SEEG pour en conclure il y a une semaine 
    que m\'opérer serait trop dangereux...\nJ\'ai des soucis de mémoire, 
    je cherche souvent mes mots, je ne retiens pas les noms de mes élèves, etc.\n
    Bref, c\'est... chiant et je suis très déçu d\'une opération impossible.\n
    Si on vous la propose, faites là !\nMoi je suis tellement déçu. 
    Je vais le tapper cette maladie toute ma vie ...'"""
    df = pipeline(txt)
    print(df)
