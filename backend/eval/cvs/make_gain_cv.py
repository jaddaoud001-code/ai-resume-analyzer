"""Génère gain_cv.pdf : un quatrième CV ENTIÈREMENT FICTIF (personne, établissements,
entreprises, coordonnées inventés ; adresse en example.com ; numéro de la plage réservée à la
fiction par l'ARCEP, 06 39 98 xx xx), développeuse backend Python, sur une seule colonne.

Conçu par l'auteur du projet pour l'évaluation de l'agent de suggestions, avec l'offre fictive
offers/logistique.txt : des faits PRÉSENTS mais mal formulés (technologies citées seulement
dans la ligne de technologies d'une expérience, SQL jamais nommé mais impliqué par PostgreSQL et
MySQL, pipeline de tests et de déploiement à chaque fusion sans le mot « CI/CD », supervision
décrite sans le mot « observabilité »), mêlés à de vraies absences (déploiement cloud, file de
messages, anglais seulement déclaré, API dont rien ne dit qu'elle est REST).

Usage (depuis backend/) :
    .venv/bin/python eval/cvs/make_gain_cv.py
"""

from pathlib import Path

import pymupdf

OUTPUT = Path(__file__).resolve().with_name("gain_cv.pdf")

A4 = pymupdf.paper_rect("a4")
MARGIN = 48

CSS = """
* { font-family: sans-serif; color: #1e293b; }
p { margin: 0 0 3px 0; font-size: 9.4px; line-height: 1.35; }
h2 { font-size: 10.5px; color: #6d28d9; margin: 10px 0 4px 0; }
.name { font-size: 20px; font-weight: bold; margin: 0 0 2px 0; }
.title { font-size: 11.5px; color: #6d28d9; margin: 0 0 4px 0; }
.contact { font-size: 9px; color: #64748b; margin: 0 0 4px 0; }
.item { font-weight: bold; font-size: 9.8px; margin: 7px 0 1px 0; }
.date { color: #64748b; font-size: 8.8px; margin: 0 0 2px 0; }
.tags { color: #475569; font-size: 8.8px; font-style: italic; margin: 1px 0 3px 0; }
.bullet { margin-left: 10px; }
.footer { font-size: 7.5px; color: #94a3b8; margin-top: 12px; }
"""

CONTENT = """
<p class="name">Inès Caradec</p>
<p class="title">Développeuse backend</p>
<p class="contact">Brest, France · +33 6 39 98 63 25 · ines.caradec@example.com</p>

<h2>PROFIL</h2>
<p>Développeuse backend depuis 2019, je construis des services web pour la logistique et le
commerce en ligne.</p>

<h2>EXPÉRIENCE PROFESSIONNELLE</h2>
<p class="item">Développeuse backend · Transports Velmaris</p>
<p class="date">Mars 2022 – en cours · Brest</p>
<p class="bullet">• Service de calcul des tarifs de livraison, appelé par le site marchand et l'application des livreurs.</p>
<p class="bullet">• Pipeline de tests et de déploiement lancé à chaque fusion sur la branche principale, sur GitLab.</p>
<p class="bullet">• Tableaux de suivi de la latence et alertes envoyées à l'astreinte en cas d'erreurs.</p>
<p class="bullet">• Traitement des requêtes lentes de la base des commandes : index et réécriture des requêtes.</p>
<p class="tags">Python · FastAPI · PostgreSQL · Docker</p>

<p class="item">Développeuse · Atelier Brumelle</p>
<p class="date">Sept. 2019 – Févr. 2022 · Quimper</p>
<p class="bullet">• Outils internes de gestion des stocks pour les entrepôts du groupe.</p>
<p class="bullet">• Reprise des scripts d'import nocturnes et de leur planification.</p>
<p class="tags">Python · Django · MySQL</p>

<h2>FORMATION</h2>
<p class="item">Licence professionnelle métiers de l'informatique · Université de la Côte des Légendes</p>
<p class="date">2018 – 2019</p>

<h2>COMPÉTENCES</h2>
<p>Langages : Python, SQL</p>
<p>Outils : FastAPI, Django, PostgreSQL, MySQL, Docker, GitLab</p>

<h2>LANGUES</h2>
<p>Français : langue maternelle · Anglais : B2</p>

<h2>CENTRES D'INTÉRÊT</h2>
<p>Course à pied, ateliers de réparation de vélos</p>

<p class="footer">CV fictif : personne, établissements, entreprises et coordonnées sont inventés.</p>
"""


def build() -> bytes:
    document = pymupdf.open()
    page = document.new_page(width=A4.width, height=A4.height)
    rect = pymupdf.Rect(MARGIN, MARGIN, A4.width - MARGIN, A4.height - MARGIN)
    spare, _ = page.insert_htmlbox(rect, CONTENT, css=CSS)
    if spare < 0:
        raise RuntimeError("Le contenu déborde de la page.")
    document.set_metadata({"title": "CV fictif (gain de reformulation)", "author": "AI Resume Analyzer"})
    # Sortie déterministe : pas de date de création ni d'identifiant aléatoire.
    return document.tobytes(garbage=4, deflate=True, no_new_id=True)


def main() -> None:
    OUTPUT.write_bytes(build())
    print(f"{OUTPUT} ({OUTPUT.stat().st_size} octets)")


if __name__ == "__main__":
    main()
