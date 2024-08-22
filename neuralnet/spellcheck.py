# Initialize Hunspell spell checker
hspell = hunspell.HunSpell(
    "/usr/share/hunspell/el_GR.dic", "/usr/share/hunspell/el_GR.aff"
)


def spell_check(text):
    corrected_text = []
    for word in text.split():
        if hspell.spell(word):
            corrected_text.append(word)
        else:
            suggestions = hspell.suggest(word)
            corrected_text.append(suggestions[0] if suggestions else word)
    return " ".join(corrected_text)


def preprocess_text(text, stopwords):
    # before tokenizing
    text = spell_check(text)
