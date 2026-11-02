"""Génère un jeu de données synthétique de tickets de support informatique en français.

Pourquoi un jeu synthétique ? Les tickets réels d'une entreprise sont confidentiels.
Le générateur reproduit les difficultés d'un vrai helpdesk pour que l'évaluation reste
honnête (le modèle ne doit pas obtenir 100 %) :
  - vocabulaire partagé entre catégories (ex. « dossier partagé » : réseau ou droits d'accès ?) ;
  - tickets qui mentionnent un deuxième sujet ;
  - fautes de frappe, absence d'accents, textes très courts ;
  - priorité souvent implicite (aucun indice d'urgence dans ~35 % des tickets) ;
  - environ 5 % d'erreurs d'étiquetage, comme lorsqu'un technicien classe mal un ticket ;
  - des « vagues » d'incidents récurrents (panne VPN, mise à jour Outlook...) à détecter.

Usage : python data/generate_dataset.py  ->  data/tickets.csv
"""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

SEED = 42
N_TICKETS = 4000
START = datetime(2026, 4, 1)
END = datetime(2026, 10, 31)

CATEGORIES = {
    "Réseau": {
        "subjects": ["Pas de connexion internet", "Wi-Fi instable", "VPN impossible", "Réseau très lent",
                     "Pas d'accès au réseau", "Connexion qui coupe", "Câble réseau défectueux"],
        "problems": [
            "je n'ai plus internet depuis ce matin sur mon poste",
            "le wifi se déconnecte toutes les dix minutes dans la salle {room}",
            "le VPN refuse de se connecter quand je travaille depuis chez moi",
            "le réseau est extrêmement lent, les pages mettent plus d'une minute à charger",
            "mon PC n'obtient pas d'adresse IP, il affiche « réseau non identifié »",
            "impossible de joindre le serveur {server}, la connexion expire",
            "la prise réseau de mon bureau ne fonctionne plus, le voyant reste éteint",
            "le partage réseau ne répond plus, l'explorateur tourne dans le vide",
            "la connexion à distance (bureau à distance) coupe sans arrêt",
            "plus de net sur le poste, ni wifi ni filaire",
        ],
    },
    "Matériel": {
        "subjects": ["PC ne démarre plus", "Écran noir", "Clavier défectueux", "Souris qui ne marche plus",
                     "Batterie HS", "Ordinateur qui chauffe", "Problème de station d'accueil"],
        "problems": [
            "mon ordinateur ne démarre plus, il y a juste un bip au lancement",
            "l'écran reste noir alors que le PC est allumé",
            "plusieurs touches du clavier ne répondent plus",
            "ma souris sans fil ne fonctionne plus malgré des piles neuves",
            "la batterie de mon portable ne tient plus que vingt minutes",
            "le PC chauffe énormément et le ventilateur fait un bruit anormal",
            "le deuxième écran n'est plus détecté sur la station d'accueil",
            "le portable s'éteint tout seul plusieurs fois par jour",
            "la webcam intégrée n'est plus reconnue",
        ],
    },
    "Logiciel": {
        "subjects": ["Application qui plante", "Erreur Excel", "Mise à jour bloquée", "Installation logiciel",
                     "Licence expirée", "ERP très lent", "Message d'erreur au lancement"],
        "problems": [
            "{app} se ferme tout seul dès que j'ouvre un fichier",
            "Excel affiche une erreur à l'ouverture des fichiers partagés",
            "la mise à jour Windows reste bloquée à {pct} % depuis hier",
            "j'ai besoin qu'on m'installe {app} pour un nouveau projet",
            "la licence de {app} a expiré, le logiciel passe en lecture seule",
            "l'ERP est très lent pour valider les commandes",
            "un message « erreur inattendue 0x{code} » apparaît au lancement de {app}",
            "le logiciel de comptabilité ne trouve plus la base de données",
            "{app} reste figé sur l'écran de chargement",
        ],
    },
    "Accès & comptes": {
        "subjects": ["Mot de passe expiré", "Compte verrouillé", "Accès refusé", "Demande de droits",
                     "Problème d'authentification", "Création de compte", "Code MFA non reçu"],
        "problems": [
            "mon mot de passe a expiré et je n'arrive pas à le changer",
            "mon compte est verrouillé après plusieurs tentatives",
            "j'ai un message « accès refusé » sur le dossier partagé de mon service",
            "j'ai besoin des droits d'accès au répertoire {folder}",
            "je ne reçois plus le code de double authentification sur mon téléphone",
            "merci de créer un compte pour un nouvel arrivant lundi dans l'équipe {team}",
            "je n'ai plus accès à l'application RH depuis mon changement de poste",
            "la session se ferme dès que je me connecte, identifiants refusés",
            "il faut retirer les accès d'un collaborateur qui a quitté l'entreprise",
            "mdp expiré, impossible de me connecter à ma session",
            "je ne peux plus me logger, identifiant ou mdp incorrect",
        ],
    },
    "Messagerie": {
        "subjects": ["Outlook ne reçoit plus", "Boîte mail pleine", "Mail bloqué", "Problème de calendrier",
                     "Teams ne fonctionne pas", "Signature mail", "Liste de diffusion"],
        "problems": [
            "Outlook ne reçoit plus aucun mail depuis ce matin",
            "ma boîte mail est pleine, je ne peux plus envoyer de messages",
            "les mails d'un client important arrivent dans les indésirables",
            "je ne vois plus le calendrier partagé de l'équipe {team}",
            "Teams se ferme pendant les réunions et je perds l'appel",
            "Outlook reste bloqué sur « synchronisation en cours »",
            "il faut ajouter un collègue à la liste de diffusion {team}",
            "ma signature mail a disparu après la mise à jour",
            "les pièces jointes de plus de 10 Mo sont refusées",
            "outlook plante à chaque envoi de mail",
        ],
    },
    "Impression": {
        "subjects": ["Imprimante hors ligne", "Bourrage papier", "Toner vide", "Scan vers mail",
                     "File d'impression bloquée", "Impression en couleur", "Nouvelle imprimante"],
        "problems": [
            "l'imprimante du {floor} étage est indiquée hors ligne",
            "bourrage papier permanent sur l'imprimante de l'accueil",
            "le toner noir est vide sur l'imprimante {printer}",
            "le scan vers mail ne fonctionne plus sur le copieur",
            "la file d'impression est bloquée, aucun document ne sort",
            "les impressions sortent avec des traits sur toute la page",
            "je n'arrive pas à ajouter l'imprimante {printer} sur mon poste",
            "l'impression en couleur est désactivée sur mon compte",
            "le copieur affiche une erreur et redémarre en boucle",
        ],
    },
}

FILL = {
    "room": ["B12", "de réunion 2", "du 3e", "Atlas", "de formation"],
    "server": ["SRV-FICHIERS", "SRV-ERP", "de production", "SAGE", "intranet"],
    "app": ["Excel", "Word", "SAP", "AutoCAD", "Sage", "Chrome", "Teams", "PowerBI"],
    "pct": ["35", "67", "88", "99"],
    "code": ["80070005", "C000021A", "8007045D", "80004005"],
    "folder": ["Comptabilité", "RH", "Projets 2026", "Achats", "Direction"],
    "team": ["commerciale", "RH", "logistique", "finance", "support"],
    "floor": ["1er", "2e", "3e", "4e"],
    "printer": ["IMP-RDC", "IMP-COMPTA", "IMP-3E", "Xerox de l'accueil"],
}

OPENERS = ["Bonjour,", "Bonjour", "Bonjour à tous,", "Salut,", "Bonjour le support,", "", "", "Hello,"]
CLOSINGS = ["Merci d'avance.", "Merci.", "Cordialement.", "Merci pour votre aide.", "", "", "Bonne journée."]
GENERIC = [
    "J'ai déjà redémarré le poste.", "Le problème a commencé après la mise à jour d'hier.",
    "Mon collègue a le même souci.", "J'ai essayé de débrancher et rebrancher.",
    "Je suis en télétravail aujourd'hui.", "Je vous joins une capture d'écran.",
    "Cela arrive une à deux fois par semaine.", "",
]
URGENCY = {
    "Critique": ["Toute l'équipe est bloquée.", "La production est à l'arrêt.", "Les clients sont impactés, c'est très urgent.",
                 "Plus personne ne peut travailler au service."],
    "Haute": ["Je suis complètement bloqué.", "C'est urgent, j'ai une réunion client dans une heure.",
              "Je ne peux pas travailler.", "Merci de traiter rapidement."],
    "Moyenne": ["Ça me ralentit beaucoup.", "Si possible dans la journée.", "Je peux contourner pour l'instant.",
                "Merci de regarder dès que possible."],
    "Basse": ["Ce n'est pas urgent.", "Quand vous aurez le temps.", "Rien de bloquant.", "Pour la semaine prochaine."],
}
PRIORITIES = ["Basse", "Moyenne", "Haute", "Critique"]
PRIORITY_WEIGHTS = [0.25, 0.40, 0.25, 0.10]
# Délai de résolution médian en heures selon la priorité.
RESOLUTION_HOURS = {"Critique": 2.5, "Haute": 5, "Moyenne": 14, "Basse": 40}
CHANNELS = ["Portail", "E-mail", "Téléphone", "Teams"]

# Vagues d'incidents récurrents : (date, catégorie, phrase, nombre de tickets)
INCIDENT_WAVES = [
    (datetime(2026, 5, 12), "Réseau", "le VPN refuse de se connecter quand je travaille depuis chez moi", 45),
    (datetime(2026, 7, 7), "Messagerie", "Outlook reste bloqué sur « synchronisation en cours »", 38),
    (datetime(2026, 9, 15), "Impression", "l'imprimante du 3e étage est indiquée hors ligne", 30),
    (datetime(2026, 10, 20), "Accès & comptes", "je ne reçois plus le code de double authentification sur mon téléphone", 35),
]


def fill(template: str, rng: random.Random) -> str:
    return template.format(**{k: rng.choice(v) for k, v in FILL.items()})


def add_typos(text: str, rng: random.Random) -> str:
    """Fautes réalistes : accents supprimés, lettre oubliée, tout en minuscules."""
    if rng.random() < 0.25:
        text = text.translate(str.maketrans("éèêàâçùûôî", "eeeaacuuoi"))
    if rng.random() < 0.20:
        words = text.split()
        i = rng.randrange(len(words))
        if len(words[i]) > 4:
            j = rng.randrange(1, len(words[i]) - 1)
            words[i] = words[i][:j] + words[i][j + 1:]
        text = " ".join(words)
    if rng.random() < 0.15:
        text = text.lower()
    return text


def make_ticket(rng: random.Random, category: str, priority: str, problem: str | None = None) -> tuple[str, str]:
    cat = CATEGORIES[category]
    subject = rng.choice(cat["subjects"])
    if rng.random() < 0.15:  # objets peu informatifs, fréquents dans la réalité
        subject = rng.choice(["Problème", "Urgent", "Help", "Demande", "Souci PC", "Question"])

    sentences = [rng.choice(OPENERS), fill(problem or rng.choice(cat["problems"]), rng).capitalize() + "."]
    if rng.random() < 0.20:  # deuxième sujet mentionné : brouille la catégorie
        other = rng.choice([c for c in CATEGORIES if c != category])
        sentences.append("Au passage, " + fill(rng.choice(CATEGORIES[other]["problems"]), rng) + ".")
    sentences.append(rng.choice(GENERIC))
    if rng.random() < 0.65:  # la priorité n'est pas toujours exprimée
        sentences.append(rng.choice(URGENCY[priority]))
    sentences.append(rng.choice(CLOSINGS))

    body = " ".join(s for s in sentences if s)
    if rng.random() < 0.10:  # ticket très court
        body = fill(problem or rng.choice(cat["problems"]), rng)
    return subject, add_typos(body, rng)


def random_datetime(rng: random.Random, day: datetime | None = None) -> datetime:
    if day is None:
        day = START + timedelta(days=rng.randrange((END - START).days + 1))
    while day.weekday() >= 5 and rng.random() < 0.85:  # peu de tickets le week-end
        day += timedelta(days=1)
    return day.replace(hour=rng.randint(8, 18), minute=rng.randint(0, 59))


def resolution(rng: random.Random, created: datetime, priority: str) -> datetime | None:
    if created > END - timedelta(days=3) or rng.random() < 0.06:
        return None  # ticket encore ouvert
    hours = rng.lognormvariate(0, 0.8) * RESOLUTION_HOURS[priority]
    return created + timedelta(hours=round(hours, 1))


def generate(n: int = N_TICKETS, seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    categories = list(CATEGORIES)

    def add(category: str, priority: str, created: datetime, problem: str | None = None) -> None:
        subject, body = make_ticket(rng, category, priority, problem)
        label = category
        if rng.random() < 0.05:  # erreur d'étiquetage humaine
            label = rng.choice([c for c in categories if c != category])
        resolved = resolution(rng, created, priority)
        rows.append({
            "created_at": created.isoformat(timespec="minutes"),
            "resolved_at": resolved.isoformat(timespec="minutes") if resolved else "",
            "channel": rng.choice(CHANNELS),
            "subject": subject,
            "body": body,
            "category": label,
            "priority": priority,
        })

    for day, category, problem, count in INCIDENT_WAVES:
        for _ in range(count):
            # Une panne générale est souvent déclarée Haute ou Critique.
            priority = rng.choices(PRIORITIES, weights=[0.05, 0.25, 0.45, 0.25])[0]
            add(category, priority, random_datetime(rng, day + timedelta(days=rng.choice([0, 0, 0, 1]))), problem)

    while len(rows) < n:
        category = rng.choices(categories, weights=[0.18, 0.15, 0.22, 0.20, 0.15, 0.10])[0]
        priority = rng.choices(PRIORITIES, weights=PRIORITY_WEIGHTS)[0]
        add(category, priority, random_datetime(rng))

    rows.sort(key=lambda r: r["created_at"])
    for i, row in enumerate(rows, start=1):
        row["ticket_id"] = f"TCK-{i:05d}"
    return rows


def main() -> None:
    rows = generate()
    out = Path(__file__).with_name("tickets.csv")
    fields = ["ticket_id", "created_at", "resolved_at", "channel", "subject", "body", "category", "priority"]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} tickets écrits dans {out}")


if __name__ == "__main__":
    main()
