"""
Parsing and cleaning logic for Wikipedia movie wikitext.
"""

import re
from typing import List, Optional, Tuple
import mwparserfromhell

def clean_actor_name(raw_name: str) -> str:
    """
    Normalizes actor names:
    - Removes Wikipedia link markup [[Actor Name|Display Name]] -> Display Name
    - Strips disambiguation tags like '(שחקן)' or '(זמר)'
    - Trims whitespace and special formatting characters
    """
    # Remove wiki link brackets if present
    parsed = mwparserfromhell.parse(raw_name)
    text = parsed.strip_code()
    
    # Strip disambiguation parentheticals e.g. "אריק איינשטיין (שחקן)" -> "אריק איינשטיין"
    text = re.sub(r'\s*\([^)]*\)', '', text)
    
    # Clean leading bullet points, HTML tags, or extra whitespace
    text = re.sub(r'[*•\-\n]', '', text)
    text = text.strip()
    
    return text

def parse_movie_metadata(wikitext: str, max_actors: int = 10) -> Tuple[Optional[int], List[str]]:
    """
    Parses release year and up to `max_actors` actors from the 'סרט' infobox template.
    Returns (year, [actor1, actor2, ...])
    """
    parsed_code = mwparserfromhell.parse(wikitext)
    templates = parsed_code.filter_templates()
    
    year: Optional[int] = None
    actors: List[str] = []

    for template in templates:
        if template.name.strip().startswith("סרט"):
            # Extract Year
            if template.has("שנת יציאה"):
                raw_year = str(template.get("שנת יציאה").value)
                match = re.search(r'\b(19\d{2}|20\d{2})\b', raw_year)
                if match:
                    year = int(match.group(1))

            # Extract Actors
            actor_fields = ["שחקנים", "שחקנים ראשיים", "כיכוב"]
            for field in actor_fields:
                if template.has(field):
                    field_val = template.get(field).value
                    # Extract internal wikilinks [[Actor Name]]
                    links = field_val.filter_wikilinks()
                    if links:
                        for link in links:
                            actor_name = clean_actor_name(str(link.title))
                            if actor_name and actor_name not in actors:
                                actors.append(actor_name)
                    else:
                        # Fallback for plain text or comma/line-separated lists
                        raw_actors = str(field_val).split("\n")
                        for line in raw_actors:
                            for piece in line.split(","):
                                cleaned = clean_actor_name(piece)
                                if cleaned and cleaned not in actors:
                                    actors.append(cleaned)

    # Fallback year search in body text if infobox year was missing
    if year is None:
        year_match = re.search(r'\b(19\d{2}|20\d{2})\b', wikitext[:1000])
        if year_match:
            year = int(year_match.group(1))

    return year, actors[:max_actors]