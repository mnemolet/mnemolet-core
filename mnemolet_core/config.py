import os
import tomllib
from pathlib import Path

CONFIG_PATH = Path(
    os.getenv("CONFIG_PATH", Path(__file__).resolve().parent.parent / "config.toml")
)


def load_config(path: Path = CONFIG_PATH):
    """
    Load configuration from TOML file.
    """
    with open(path, "rb") as f:
        return tomllib.load(f)


config = load_config()

QDRANT_HOST = os.getenv("QDRANT_HOST", config["qdrant"]["host"])
QDRANT_PORT = os.getenv("QDRANT_PORT", config["qdrant"]["port"])
QDRANT_COLLECTION = config["qdrant"]["collection"]
QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"

EMBED_MODEL = os.getenv("EMBED_MODEL", config["embedding"]["model"])
EMBED_BATCH = os.getenv("EMBED_BATCH", config["embedding"]["batch_size"])

OLLAMA_HOST = os.getenv("OLLAMA_HOST", config["ollama"]["host"])
OLLAMA_PORT = os.getenv("OLLAMA_PORT", config["ollama"]["port"])
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

DB_PATH = Path(os.path.expanduser(config["storage"]["db_path"]))
