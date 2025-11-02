import tomllib
from pathlib import Path
import os

CONFIG_PATH = Path(
    os.getenv("CONFIG_PATH", Path(__file__).resolve().parent.parent / "config.toml")
)


def load_config(path: Path = CONFIG_PATH):
    """
    Load configuration from TOML file.
    """
    print(path)
    with open(path, "rb") as f:
        return tomllib.load(f)


config = load_config()

QDRANT_HOST = os.getenv("QDRANT_HOST", config["qdrant"]["host"])
QDRANT_PORT = os.getenv("QDRANT_PORT", config["qdrant"]["port"])
QDRANT_COLLECTION = config["qdrant"]["collection"]
QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"

EMBED_MODEL = os.getenv("EMBED_MODEL", config["embedding"]["model"])
EMBED_BATCH = os.getenv("EMBED_BATCH", config["embedding"]["batch_size"])

DB_PATH = Path(os.path.expanduser(config["storage"]["db_path"]))
