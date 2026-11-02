from ticketsense.preprocessing import clean_text, strip_accents, ticket_text


def test_strip_accents():
    assert strip_accents("Écran bloqué à l'accueil") == "Ecran bloque a l'accueil"


def test_clean_text_normalise_casse_ponctuation_espaces():
    assert clean_text("  Outlook   ne reçoit PLUS !!! ") == "outlook ne recoit plus"


def test_clean_text_vide():
    assert clean_text(None) == ""
    assert clean_text("") == ""


def test_ticket_text_combine_objet_et_corps():
    assert ticket_text("VPN", "Impossible de se connecter") == "vpn impossible de se connecter"
    assert ticket_text(None, "Toner vide") == "toner vide"
