"""Génère backend_cv.pdf : un troisième CV ENTIÈREMENT FICTIF (personne, établissements,
entreprises, coordonnées inventés ; adresse en example.com ; numéro de la plage réservée à
la fiction par l'ARCEP, 06 39 98 xx xx), développeuse backend Python, sur une seule colonne.

Conçu par l'auteur du projet pour les « cas difficiles » de l'évaluation du matching, avec
l'offre fictive offers/logistique.txt : preuves dispersées sur plusieurs expériences (durées
à additionner, chevauchement de deux périodes), compétences prouvées sans être nommées
comme dans l'offre, première expérience assez longue pour être découpée en deux chunks.

Usage (depuis backend/) :
    .venv/bin/python eval/cvs/make_backend_cv.py
"""

from pathlib import Path

import pymupdf

OUTPUT = Path(__file__).resolve().with_name("backend_cv.pdf")

A4 = pymupdf.paper_rect("a4")
MARGIN = 46

CSS = """
* { font-family: sans-serif; color: #1c1917; }
p { margin: 0 0 3px 0; font-size: 9.2px; line-height: 1.35; }
h2 { font-size: 10.5px; color: #115e59; margin: 9px 0 4px 0; }
.name { font-size: 20px; font-weight: bold; margin: 0 0 2px 0; }
.title { font-size: 11.5px; color: #115e59; margin: 0 0 4px 0; }
.contact { font-size: 9px; color: #57534e; margin: 0 0 4px 0; }
.item { font-weight: bold; font-size: 9.8px; margin: 7px 0 1px 0; }
.date { color: #78716c; font-size: 8.8px; margin: 0 0 2px 0; }
.tags { color: #57534e; font-size: 8.8px; font-style: italic; margin: 1px 0 3px 0; }
.bullet { margin-left: 10px; }
.footer { font-size: 7.5px; color: #a8a29e; margin-top: 10px; }
"""

CONTENT = """
<p class="name">Léna Morvan</p>
<p class="title">Développeuse Python · services web</p>
<p class="contact">Nantes, France · +33 6 39 98 51 08 · lena.morvan@example.com</p>

<h2>PROFIL</h2>
<p>Développeuse orientée produit, j'aime expliquer simplement des choix techniques aux
équipes métier. Je cherche un poste où les services que je construis servent directement
leurs utilisateurs.</p>

<h2>EXPÉRIENCE PROFESSIONNELLE</h2>
<p class="item">Ingénieure logicielle · Ferlane Fret</p>
<p class="date">Févr. 2025 – en cours · Nantes</p>
<p class="bullet">• Développement des endpoints FastAPI consommés par l'application mobile des chauffeurs : authentification, suivi des tournées, notifications de livraison.</p>
<p class="bullet">• Mise en production des services sur un cluster Kubernetes managé : images Docker, manifestes Helm, montée en charge automatique selon le trafic.</p>
<p class="bullet">• Pipelines GitHub Actions : tests, analyse statique et déploiement à chaque fusion sur la branche principale.</p>
<p class="bullet">• Tests pytest sur les endpoints critiques ; couverture portée de 45 % à 80 %.</p>
<p class="bullet">• Tableaux de bord Grafana et alertes Prometheus sur la latence et le taux d'erreurs des services.</p>
<p class="bullet">• Événements de suivi transmis par RabbitMQ (mars – août 2025), puis migration vers Kafka (juin 2025 – janv. 2026).</p>
<p class="bullet">• Points hebdomadaires en anglais avec l'équipe d'exploitation de Toronto ; documentation d'architecture rédigée en anglais.</p>
<p class="tags">FastAPI · PostgreSQL · Kubernetes · Kafka</p>

<p class="item">Développeuse freelance · missions pour des PME</p>
<p class="date">Janv. 2023 – Déc. 2023 · Nantes</p>
<p class="bullet">• Applications web Django pour trois clients (gestion de stock, réservations, facturation), seule du recueil des besoins jusqu'à la livraison chez le client.</p>
<p class="bullet">• Bases PostgreSQL et interfaces d'échange avec les logiciels comptables des clients.</p>
<p class="tags">Django · PostgreSQL · Docker</p>

<p class="item">Développeuse Python (données) · Observatoire ligérien des mobilités</p>
<p class="date">Sept. 2021 – Déc. 2022 · Angers</p>
<p class="bullet">• Scripts Python d'import et de nettoyage des relevés de trafic, exécutés chaque nuit.</p>
<p class="bullet">• Requêtes SQL d'agrégation pour les rapports mensuels.</p>
<p class="tags">Python · Pandas · SQL</p>

<h2>FORMATION</h2>
<p class="item">Master informatique, parcours génie logiciel · Université de la Côte d'Émeraude</p>
<p class="date">2019 – 2021</p>

<h2>COMPÉTENCES</h2>
<p>Langages : Python, SQL, Go (notions)</p>
<p>Outils : FastAPI, Django, PostgreSQL, Docker, Kubernetes, RabbitMQ, Kafka</p>

<h2>LANGUES</h2>
<p>Français : langue maternelle · Anglais : courant</p>

<h2>CENTRES D'INTÉRÊT</h2>
<p>Voile, chorale, fabrication de claviers mécaniques</p>

<p class="footer">CV fictif : personne, établissements, entreprises et coordonnées sont inventés.</p>
"""


def build() -> bytes:
    document = pymupdf.open()
    page = document.new_page(width=A4.width, height=A4.height)
    rect = pymupdf.Rect(MARGIN, MARGIN, A4.width - MARGIN, A4.height - MARGIN)
    spare, _ = page.insert_htmlbox(rect, CONTENT, css=CSS)
    if spare < 0:
        raise RuntimeError("Le contenu déborde de la page.")
    document.set_metadata({"title": "CV fictif (backend Python)", "author": "AI Resume Analyzer"})
    # Sortie déterministe : pas de date de création ni d'identifiant aléatoire.
    return document.tobytes(garbage=4, deflate=True, no_new_id=True)


def main() -> None:
    OUTPUT.write_bytes(build())
    print(f"{OUTPUT} ({OUTPUT.stat().st_size} octets)")


if __name__ == "__main__":
    main()
