"""Génère demo_cv.pdf : un CV ENTIÈREMENT FICTIF (personne, école, entreprises, coordonnées
inventées ; adresses en example.com ; numéro de téléphone de la plage réservée à la fiction
par l'ARCEP, 06 39 98 xx xx), sur une page et deux colonnes, pour les démos, les captures
d'écran et le test de l'extraction PDF (ordre de lecture des colonnes, puces, sections).

Usage (depuis backend/) :
    .venv/bin/python eval/cvs/make_demo_cv.py
"""

from pathlib import Path

import pymupdf

OUTPUT = Path(__file__).resolve().with_name("demo_cv.pdf")

A4 = pymupdf.paper_rect("a4")
MARGIN = 36
HEADER_HEIGHT = 92
SIDEBAR_WIDTH = 178
GUTTER = 22

CSS = """
* { font-family: sans-serif; color: #1f2937; }
p { margin: 0 0 3px 0; font-size: 9px; line-height: 1.35; }
h2 { font-size: 10.5px; color: #0f4c81; margin: 9px 0 4px 0; letter-spacing: 0.5px; }
.name { font-size: 22px; font-weight: bold; color: #0f172a; margin: 0 0 2px 0; }
.title { font-size: 11.5px; color: #0f4c81; margin: 0 0 6px 0; }
.contact { font-size: 9px; color: #475569; }
.item { font-weight: bold; font-size: 9.5px; margin: 6px 0 1px 0; }
.date { color: #64748b; font-size: 8.5px; margin: 0 0 2px 0; }
.tags { color: #475569; font-size: 8.5px; font-style: italic; margin: 1px 0 3px 0; }
.bullet { margin-left: 8px; }
.footer { font-size: 7.5px; color: #94a3b8; }
"""

HEADER = """
<p class="name">Camille Lefèvre</p>
<p class="title">Développeuse IA junior · Python, LLM, RAG</p>
<p class="contact">Nantes, France · +33 6 39 98 42 17 · camille.lefevre@example.com · https://camille-lefevre.example.com</p>
"""

SIDEBAR = """
<h2>COMPÉTENCES</h2>
<p>Python, TypeScript, SQL</p>
<p>LLM, RAG, embeddings, évaluation de modèles</p>
<p>scikit-learn, PyTorch, Pandas</p>
<p>FastAPI, PostgreSQL, FAISS</p>
<p>Docker, Git, GitLab CI, Linux</p>

<h2>LANGUES</h2>
<p>Français : langue maternelle</p>
<p>Anglais : B2 (TOEIC 905)</p>
<p>Espagnol : A2</p>

<h2>CENTRES D'INTÉRÊT</h2>
<p>Escalade, soutien scolaire bénévole, photographie argentique</p>
"""

MAIN = """
<h2>PROFIL</h2>
<p>Ingénieure en informatique et IA, je conçois des assistants fondés sur des LLM, de la
préparation des données à l'évaluation, et je les mets entre les mains de leurs
utilisateurs.</p>

<h2>EXPÉRIENCE PROFESSIONNELLE</h2>
<p class="item">Stage ingénieure IA · Lumen Analytique</p>
<p class="date">Févr. 2026 – Août 2026 · Nantes</p>
<p class="bullet">• Assistant de recherche documentaire (RAG) sur 12 000 fiches internes : découpage, embeddings multilingues, index FAISS.</p>
<p class="bullet">• Jeu d'évaluation de 200 questions ; comparaison de trois LLM sur l'exactitude, le coût et la latence.</p>
<p class="bullet">• API FastAPI, tests d'intégration et déploiement Docker sur le serveur de l'équipe.</p>
<p class="tags">Python · FastAPI · FAISS · Docker</p>

<p class="item">Alternance développeuse data · Coopérative Ardoise</p>
<p class="date">Sept. 2024 – Janv. 2026 · Rennes</p>
<p class="bullet">• Tableaux de bord de suivi des demandes de support en SQL et Python ; rapports hebdomadaires automatisés.</p>
<p class="bullet">• Nettoyage et pseudonymisation des données clients avant analyse.</p>
<p class="tags">SQL · Pandas · PostgreSQL</p>

<h2>PROJETS</h2>
<p class="item">Classifieur de tickets de support</p>
<p class="date">2025 · projet d'école</p>
<p class="bullet">• Classement de 5 000 tickets par catégorie et priorité, avec scikit-learn puis un LLM ; précision 0,87.</p>
<p class="tags">Python · scikit-learn · LLM</p>

<h2>FORMATION</h2>
<p class="item">Diplôme d'ingénieure, informatique et IA · École d'ingénieurs de la Côte d'Opale</p>
<p class="date">2021 – 2026</p>
<p class="item">Classes préparatoires MPSI / MP · Lycée des Embruns</p>
<p class="date">2019 – 2021</p>

<p class="footer">CV fictif de démonstration : personne, établissements, entreprises et coordonnées sont inventés.</p>
"""


def build() -> bytes:
    document = pymupdf.open()
    page = document.new_page(width=A4.width, height=A4.height)
    top = MARGIN + HEADER_HEIGHT
    # Bandeau de la colonne de gauche.
    page.draw_rect(
        pymupdf.Rect(0, top - 8, MARGIN + SIDEBAR_WIDTH + GUTTER / 2, A4.height),
        color=None,
        fill=(0.95, 0.96, 0.98),
    )
    boxes = [
        (pymupdf.Rect(MARGIN, MARGIN, A4.width - MARGIN, top - 12), HEADER),
        (pymupdf.Rect(MARGIN, top, MARGIN + SIDEBAR_WIDTH, A4.height - MARGIN), SIDEBAR),
        (pymupdf.Rect(MARGIN + SIDEBAR_WIDTH + GUTTER, top, A4.width - MARGIN, A4.height - MARGIN), MAIN),
    ]
    for rect, html in boxes:
        spare, _ = page.insert_htmlbox(rect, html, css=CSS)
        if spare < 0:
            raise RuntimeError(f"Le contenu déborde de sa colonne ({rect}).")
    document.set_metadata({"title": "CV fictif de démonstration", "author": "AI Resume Analyzer"})
    # Sortie déterministe : pas de date de création ni d'identifiant aléatoire.
    return document.tobytes(garbage=4, deflate=True, no_new_id=True)


def main() -> None:
    OUTPUT.write_bytes(build())
    print(f"{OUTPUT} ({OUTPUT.stat().st_size} octets)")


if __name__ == "__main__":
    main()
