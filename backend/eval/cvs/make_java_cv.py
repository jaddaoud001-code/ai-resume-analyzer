"""Génère java_cv.pdf : un second CV ENTIÈREMENT FICTIF (personne, établissements,
entreprises, coordonnées inventés ; adresse en example.com ; numéro de la plage réservée à
la fiction par l'ARCEP, 06 39 98 xx xx), développeur backend Java junior, sur une seule
colonne (le premier, demo_cv.pdf, en a deux). Sert à l'évaluation du matching
(eval/eval_matching.py) : profil éloigné des offres d'IA de référence, sans LLM ni IA.

Usage (depuis backend/) :
    .venv/bin/python eval/cvs/make_java_cv.py
"""

from pathlib import Path

import pymupdf

OUTPUT = Path(__file__).resolve().with_name("java_cv.pdf")

A4 = pymupdf.paper_rect("a4")
MARGIN = 48

CSS = """
* { font-family: serif; color: #222222; }
p { margin: 0 0 3px 0; font-size: 9.5px; line-height: 1.35; }
h2 { font-size: 11px; color: #7c2d12; margin: 10px 0 4px 0; border-bottom: 1px solid #d6d3d1; }
.name { font-size: 20px; font-weight: bold; margin: 0 0 2px 0; }
.title { font-size: 12px; color: #7c2d12; margin: 0 0 4px 0; }
.contact { font-size: 9px; color: #57534e; margin: 0 0 4px 0; }
.item { font-weight: bold; font-size: 10px; margin: 7px 0 1px 0; }
.date { color: #78716c; font-size: 9px; margin: 0 0 2px 0; }
.tags { color: #57534e; font-size: 9px; font-style: italic; margin: 1px 0 3px 0; }
.bullet { margin-left: 10px; }
.footer { font-size: 7.5px; color: #a8a29e; margin-top: 12px; }
"""

CONTENT = """
<p class="name">Bastien Corlay</p>
<p class="title">Développeur backend Java junior</p>
<p class="contact">Angers, France · +33 6 39 98 27 64 · bastien.corlay@example.com</p>

<h2>PROFIL</h2>
<p>Développeur backend Java, deux ans d'alternance sur des API Spring Boot et des échanges
de données entre applications. Je cherche un premier poste en CDI dans une équipe backend.</p>

<h2>EXPÉRIENCE PROFESSIONNELLE</h2>
<p class="item">Alternance développeur backend Java · Valmorin Logistique</p>
<p class="date">Sept. 2023 – Août 2025 · Angers</p>
<p class="bullet">• Développement d'une API REST de suivi des expéditions avec Spring Boot et JPA.</p>
<p class="bullet">• Requêtes SQL et index sur PostgreSQL pour accélérer les recherches de colis.</p>
<p class="bullet">• Échanges avec les transporteurs au format XML : génération avec JAXB et validation par schéma XSD.</p>
<p class="bullet">• Rédaction de la documentation technique de l'API (OpenAPI) et d'un guide d'exploitation.</p>
<p class="tags">Java 17 · Spring Boot · PostgreSQL · Maven</p>

<p class="item">Stage développeur · Atelier Numérique Brisecôte</p>
<p class="date">Avr. 2023 – Juin 2023 · Saumur</p>
<p class="bullet">• Maintenance d'une application Java EE : correction d'anomalies et tests JUnit.</p>
<p class="bullet">• Travail en équipe de quatre avec Git : branches, revues de merge requests sur GitLab.</p>
<p class="tags">Java · JUnit · Git</p>

<h2>PROJETS</h2>
<p class="item">Gestion de prêts pour une médiathèque</p>
<p class="date">2024 · projet d'IUT</p>
<p class="bullet">• Application web Spring Boot et Thymeleaf, base MySQL ; gestion des réservations et des retards.</p>
<p class="tags">Java · Spring Boot · MySQL</p>

<h2>FORMATION</h2>
<p class="item">BUT Informatique, parcours développement · IUT du Val d'Ysère</p>
<p class="date">2022 – 2025</p>
<p class="item">Baccalauréat général, spécialités mathématiques et NSI · Lycée des Ardoisières</p>
<p class="date">2022</p>

<h2>COMPÉTENCES</h2>
<p>Langages : Java, SQL, XML, notions de Python</p>
<p>Frameworks et outils : Spring Boot, JPA / Hibernate, Maven, JUnit, Git, Docker (notions), Linux</p>

<h2>LANGUES</h2>
<p>Français : langue maternelle · Anglais : B1 (TOEIC 785) · Allemand : notions</p>

<h2>CENTRES D'INTÉRÊT</h2>
<p>Course d'orientation, club de jeux de société, bricolage électronique</p>

<p class="footer">CV fictif : personne, établissements, entreprises et coordonnées sont inventés.</p>
"""


def build() -> bytes:
    document = pymupdf.open()
    page = document.new_page(width=A4.width, height=A4.height)
    rect = pymupdf.Rect(MARGIN, MARGIN, A4.width - MARGIN, A4.height - MARGIN)
    spare, _ = page.insert_htmlbox(rect, CONTENT, css=CSS)
    if spare < 0:
        raise RuntimeError("Le contenu déborde de la page.")
    document.set_metadata({"title": "CV fictif (Java)", "author": "AI Resume Analyzer"})
    # Sortie déterministe : pas de date de création ni d'identifiant aléatoire.
    return document.tobytes(garbage=4, deflate=True, no_new_id=True)


def main() -> None:
    OUTPUT.write_bytes(build())
    print(f"{OUTPUT} ({OUTPUT.stat().st_size} octets)")


if __name__ == "__main__":
    main()
