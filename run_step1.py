import json
import os
import logging
from src.scraper import get_category_members, fetch_page_wikitext
from src.parser import parse_movie_metadata
from src.graph_builder import build_co_star_graph, export_graph_data

RAW_DATA_PATH = "data/raw/raw_movies.json"
PROCESSED_CSV_PATH = "data/processed/israeli_actors_edges.csv"
GEXF_PATH = "data/processed/israeli_actors_graph.gexf"

def main():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # 1. Scraping / Fetching Raw Data
    if not os.path.exists(RAW_DATA_PATH):
        members = get_category_members("קטגוריה:סרטים ישראליים")
        movie_data = []

        logging.info("Downloading wikitext for movie pages...")
        for idx, member in enumerate(members, 1):
            page_id = member["pageid"]
            title = member["title"]
            logging.info(f"[{idx}/{len(members)}] Processing: {title}")
            
            try:
                wikitext = fetch_page_wikitext(page_id)
                movie_data.append({
                    "pageid": page_id,
                    "title": title,
                    "wikitext": wikitext
                })
            except Exception as e:
                logging.error(f"Failed to fetch {title}: {e}")

        with open(RAW_DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(movie_data, f, ensure_ascii=False, indent=2)
    else:
        logging.info(f"Loading cached raw data from {RAW_DATA_PATH}")
        with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
            movie_data = json.load(f)

    # 2. Parsing Metadata & Actor Extraction
    parsed_records = []
    for item in movie_data:
        year, actors = parse_movie_metadata(item["wikitext"], max_actors=10)
        if year and len(actors) >= 2:
            parsed_records.append({
                "title": item["title"],
                "year": year,
                "actors": actors
            })

    logging.info(f"Successfully extracted cast records for {len(parsed_records)} movies.")

    # 3. Building Graph & Exporting
    G = build_co_star_graph(parsed_records)
    export_graph_data(G, PROCESSED_CSV_PATH, GEXF_PATH)

if __name__ == "__main__":
    main()