"""
MediaWiki API scraper for Israeli movies category on Hebrew Wikipedia.
"""

import logging
import time
from typing import Dict, List, Any
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

API_URL = "https://he.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "IsraeliCinemaGraphProject/1.0 (academic_research@university.ac.il)"
}

def get_category_members(category_title: str = "קטגוריה:סרטים ישראליים") -> List[Dict[str, Any]]:
    """
    Fetches all pages directly belonging to the specified Wikipedia category.
    """
    pages = []
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category_title,
        "cmlimit": "max",
        "cmtype": "page",
        "format": "json"
    }

    logging.info(f"Fetching category members for {category_title}...")
    while True:
        response = requests.get(API_URL, headers=HEADERS, params=params)
        response.raise_for_status()
        data = response.json()
        
        members = data.get("query", {}).get("categorymembers", [])
        pages.extend(members)

        if "continue" in data:
            params["cmcontinue"] = data["continue"]["cmcontinue"]
            time.sleep(0.2)  # API rate limit courtesy
        else:
            break

    logging.info(f"Retrieved {len(pages)} movie pages from category.")
    return pages

def fetch_page_wikitext(page_id: int) -> str:
    """
    Retrieves the raw wikitext of a given Wikipedia page ID.
    """
    params = {
        "action": "parse",
        "pageid": page_id,
        "prop": "wikitext",
        "format": "json"
    }
    response = requests.get(API_URL, headers=HEADERS, params=params)
    response.raise_for_status()
    data = response.json()
    return data.get("parse", {}).get("wikitext", {}).get("*", "")