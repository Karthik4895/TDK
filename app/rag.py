import logging
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger("tdk.rag")

_DEFAULT_DOCUMENTS = [
    # ── Products ──
    "Organic Henna Cones from Mehandi Tales: pack of 5 cones each 20gms, costs Rs 60 only",
    "Combo pack from Mehandi Tales includes organic henna cones plus after care oil, priced at Rs 299",
    "Bridal Kit from Mehandi Tales contains premium cones and aftercare essentials, costs Rs 499",
    "We sell three products: Organic Henna Cones (Rs 60 for 5 cones), Combo Pack with oil (Rs 299), and Bridal Kit (Rs 499)",
    "Mehandi Tales products can be ordered via WhatsApp for home delivery",
    "After care oil from Mehandi Tales helps maintain mehendi color and keeps skin moisturized",

    # ── Services ──
    "Basic mehandi design on both hands palms only costs Rs 500 at Mehandi Tales By Divya",
    "3/4 coverage both hands basic mehandi design service costs Rs 1000 at Mehandi Tales",
    "3/4 coverage both hands bridal mehandi design costs Rs 2000 at Mehandi Tales By Divya",
    "Full bridal mehandi design on both hands complete coverage costs Rs 3500 at Mehandi Tales",
    "Full bridal mehandi design on both legs costs Rs 1500 at Mehandi Tales By Divya",
    "Mehandi Tales By Divya offers home visits for mehendi application across Chennai",
    "Service pricing: Basic palms Rs 500, 3/4 basic Rs 1000, 3/4 bridal Rs 2000, full hands Rs 3500, full legs Rs 1500",
    "Book mehandi services from Mehandi Tales By Divya via WhatsApp for appointments",

    # ── General & Tips ──
    "Mehandi Tales By Divya is run by Divya, a professional mehendi artist in Chennai",
    "All henna products at Mehandi Tales are 100% organic, chemical-free and safe for all skin types",
    "Bridal bookings at Mehandi Tales should be done at least 2 weeks in advance",
    "The best time to apply bridal mehendi is 1 to 2 days before the wedding ceremony",
    "Mehendi takes 2 to 4 hours to dry completely; keeping it on longer gives deeper color",
    "Apply lemon juice and sugar mixture on dry mehendi to enhance and deepen the color",
    "Organic henna gives rich dark brown to reddish color that darkens overnight",
    "Arabic mehendi designs are simple elegant and perfect for all occasions",
    "Rajasthani bridal mehendi features intricate peacock and paisley motifs",
    "Indo-Arabic fusion mehendi blends traditional Indian patterns with modern Arabic style",
    "Patch test recommended for sensitive skin customers before full application",
    "Glitter and stone embellishments can be added to mehendi designs on request",
    "Mehandi Tales offers group bookings for corporate events and kitty parties at discounted rates",
    "Full hand bridal mehendi takes 3 to 5 hours for Divya to complete",
    "Cone mehendi applicator is used for detailed and fine-line professional designs",
]


def _load_from_data_dir() -> list[str]:
    docs = list(_DEFAULT_DOCUMENTS)
    data_dir = Path("data")
    if data_dir.exists():
        for txt_file in data_dir.glob("*.txt"):
            try:
                lines = [l.strip() for l in txt_file.read_text(encoding="utf-8").splitlines() if l.strip()]
                docs.extend(lines)
                logger.info("Loaded %d lines from %s", len(lines), txt_file.name)
            except Exception as exc:
                logger.warning("Could not load %s: %s", txt_file, exc)
    return docs


# Runs locally — no API key needed. Downloads ~90 MB on first run.
_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
_documents = _load_from_data_dir()
_vectorstore = FAISS.from_texts(_documents, _embedding)
logger.info("Vector store ready with %d documents", len(_documents))


def retrieve(query: str, k: int = 4) -> str:
    try:
        docs = _vectorstore.similarity_search(query, k=k)
        return "\n".join(d.page_content for d in docs)
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        return ""


def add_documents(texts: list[str]):
    _vectorstore.add_texts(texts)
    logger.info("Added %d documents to vector store", len(texts))
