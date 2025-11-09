import logging

import requests
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)


def check_qdrant_status(url: str, endpoint: str = "healthz") -> bool:
    """
    Check if Qdrant is alive, ready, or healthy via its HTTP status endpoint.

    Return bool

    Reference:
        https://qdrant.tech/documentation/guides/monitoring/
    """
    try:
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            logger.info(f"Qdrant {endpoint} check passed at {url}")
            return True
        else:
            logger.warning(f"Unexpected response from {url}: {response.text}")
            return False
    except RequestException as e:
        logger.error(f"Could not connect to Qdrant at {url}: {e}")
        return False
