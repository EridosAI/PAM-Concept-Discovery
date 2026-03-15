"""Download science fiction novels from Project Gutenberg.

Queries the Gutendex API for English-language science fiction novels,
downloads plain-text versions, strips Gutenberg boilerplate, and saves
cleaned texts alongside a metadata manifest.

Usage:
    python 01_download_corpus/download_gutenberg.py
    python 01_download_corpus/download_gutenberg.py --max 50
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
import re
import json
import time
import argparse
import requests
from tqdm import tqdm

from utils.config import Config

GUTENDEX_API = "https://gutendex.com/books/"
GUTENBERG_MIRROR = "https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"

# Curated seed list of known Gutenberg sci-fi IDs to guarantee coverage.
# These are reliably sci-fi and may not all surface via API subject search.
# fmt: off
SEED_SCIFI = [
    # --- H.G. Wells ---
    (35,    "The Time Machine",              "H.G. Wells"),
    (36,    "The War of the Worlds",         "H.G. Wells"),
    (159,   "The Island of Doctor Moreau",   "H.G. Wells"),
    (5230,  "The Invisible Man",             "H.G. Wells"),
    (718,   "The First Men in the Moon",     "H.G. Wells"),
    (6927,  "The Sleeper Awakes",            "H.G. Wells"),
    (1743,  "The Food of the Gods",          "H.G. Wells"),
    (34962, "In the Days of the Comet",      "H.G. Wells"),
    (44852, "The World Set Free",            "H.G. Wells"),
    (27706, "Men Like Gods",                 "H.G. Wells"),
    # --- Jules Verne ---
    (164,   "Twenty Thousand Leagues Under the Sea", "Jules Verne"),
    (863,   "The Mysterious Island",         "Jules Verne"),
    (18857, "From the Earth to the Moon",    "Jules Verne"),
    (3748,  "A Journey to the Centre of the Earth", "Jules Verne"),
    (83,    "Five Weeks in a Balloon",       "Jules Verne"),
    (1268,  "Off on a Comet",               "Jules Verne"),
    (3808,  "Robur the Conqueror",           "Jules Verne"),
    # --- Edgar Rice Burroughs (Mars/Pellucidar) ---
    (62,    "A Princess of Mars",            "Edgar Rice Burroughs"),
    (64,    "The Gods of Mars",              "Edgar Rice Burroughs"),
    (68,    "Warlord of Mars",               "Edgar Rice Burroughs"),
    (72,    "Thuvia, Maid of Mars",          "Edgar Rice Burroughs"),
    (1153,  "At the Earth's Core",           "Edgar Rice Burroughs"),
    (605,   "Pellucidar",                    "Edgar Rice Burroughs"),
    (119,   "The Chessmen of Mars",          "Edgar Rice Burroughs"),
    # --- Mary Shelley ---
    (84,    "Frankenstein",                  "Mary Shelley"),
    (18247, "The Last Man",                  "Mary Shelley"),
    # --- E.E. "Doc" Smith ---
    (20869, "The Skylark of Space",          "E.E. Smith"),
    (20796, "Spacehounds of IPC",            "E.E. Smith"),
    (32706, "Triplanetary",                  "E.E. Smith"),
    (21814, "First Lensman",                 "E.E. Smith"),
    (24693, "Galactic Patrol",               "E.E. Smith"),
    (23731, "Gray Lensman",                  "E.E. Smith"),
    (25024, "Second Stage Lensmen",          "E.E. Smith"),
    # --- H. Beam Piper ---
    (18137, "Little Fuzzy",                  "H. Beam Piper"),
    (19370, "Space Viking",                  "H. Beam Piper"),
    (18346, "Uller Uprising",                "H. Beam Piper"),
    (20726, "Omnilingual",                   "H. Beam Piper"),
    (18768, "Naudsonce",                     "H. Beam Piper"),
    (22393, "The Cosmic Computer",           "H. Beam Piper"),
    # --- Murray Leinster ---
    (32154, "The Wailing Asteroid",          "Murray Leinster"),
    (29728, "The Pirates of Ersatz",         "Murray Leinster"),
    (18346, "Operation Terror",              "Murray Leinster"),
    (30461, "The Greks Bring Gifts",         "Murray Leinster"),
    # --- Andre Norton ---
    (18346, "Star Born",                     "Andre Norton"),
    (18846, "The Defiant Agents",            "Andre Norton"),
    (19471, "Voodoo Planet",                 "Andre Norton"),
    (20669, "The Time Traders",              "Andre Norton"),
    (21081, "Storm Over Warlock",            "Andre Norton"),
    # --- Philip K. Dick (public domain works) ---
    (32522, "Beyond the Door",               "Philip K. Dick"),
    (28698, "The Eyes Have It",              "Philip K. Dick"),
    (32154, "Second Variety",                "Philip K. Dick"),
    (28644, "The Variable Man",              "Philip K. Dick"),
    (32832, "The Skull",                     "Philip K. Dick"),
    (28767, "Mr. Spaceship",                 "Philip K. Dick"),
    # --- Isaac Asimov (public domain) ---
    (31547, "Youth",                         "Isaac Asimov"),
    # --- Robert Heinlein (public domain) ---
    (17246, "The Green Hills of Earth",      "Robert A. Heinlein"),
    # --- Poul Anderson ---
    (24196, "The Burning Bridge",            "Poul Anderson"),
    (25050, "The Chapter Ends",              "Poul Anderson"),
    (30098, "Duel on Syrtis",                "Poul Anderson"),
    (29523, "Tiger by the Tail",             "Poul Anderson"),
    # --- Clifford D. Simak ---
    (29579, "Immigrant",                     "Clifford D. Simak"),
    # --- Frederik Pohl ---
    (51233, "The Tunnel Under the World",    "Frederik Pohl"),
    # --- Edmond Hamilton ---
    (64937, "The Star Kings",                "Edmond Hamilton"),
    (60335, "The World With a Thousand Moons", "Edmond Hamilton"),
    # --- Jack Williamson ---
    (29567, "The Cosmic Express",            "Jack Williamson"),
    # --- Stanley G. Weinbaum ---
    (22893, "A Martian Odyssey",             "Stanley G. Weinbaum"),
    # --- Ray Cummings ---
    (21094, "The Girl in the Golden Atom",   "Ray Cummings"),
    (62, "Brigands of the Moon",             "Ray Cummings"),
    # --- Kurt Vonnegut (early) ---
    (30240, "Harrison Bergeron",             "Kurt Vonnegut"),
    # --- Other classic sci-fi ---
    (624,   "Looking Backward",              "Edward Bellamy"),
    (1607,  "A Modern Utopia",               "H.G. Wells"),
    (2621,  "The Coming Race",               "Edward Bulwer-Lytton"),
    (1906,  "Erewhon",                       "Samuel Butler"),
    (5765,  "Caesar's Column",               "Ignatius Donnelly"),
    (6424,  "The Iron Heel",                 "Jack London"),
    (10071, "The Star Rover",                "Jack London"),
    (1015,  "The Scarlet Plague",            "Jack London"),
    (1080,  "A Connecticut Yankee",          "Mark Twain"),
    (620,   "Herland",                       "Charlotte Perkins Gilman"),
    (3600,  "Gulliver's Travels",            "Jonathan Swift"),
]
# fmt: on


def query_gutendex_scifi(max_pages: int = 20) -> list[dict]:
    """Query Gutendex API for English science fiction novels."""
    books = []
    url = f"{GUTENDEX_API}?topic=science+fiction&languages=en&mime_type=text/plain"

    for _ in range(max_pages):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  [WARN] Gutendex query failed: {e}")
            break

        for book in data.get("results", []):
            books.append({
                "id": book["id"],
                "title": book.get("title", "Unknown"),
                "authors": [a.get("name", "Unknown") for a in book.get("authors", [])],
                "download_count": book.get("download_count", 0),
            })

        url = data.get("next")
        if not url:
            break
        time.sleep(0.5)

    return books


def download_book(book_id: int, cache_dir: str) -> str | None:
    """Download a single book from Gutenberg, caching to disk."""
    cache_path = os.path.join(cache_dir, f"pg{book_id}.txt")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    url = GUTENBERG_MIRROR.format(book_id=book_id)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        text = resp.text
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(text)
        return text
    except Exception as e:
        print(f"  [WARN] Failed to download book {book_id}: {e}")
        return None


def clean_gutenberg_text(text: str) -> str:
    """Remove Project Gutenberg header/footer boilerplate."""
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
        "*END*THE SMALL PRINT",
    ]
    start_idx = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            nl = text.find("\n", idx)
            if nl != -1:
                start_idx = nl + 1
            break

    end_markers = [
        "*** END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
    ]
    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = idx
            break

    cleaned = text[start_idx:end_idx].strip()
    cleaned = re.sub(r"\r\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def count_words(text: str) -> int:
    return len(text.split())


def run(config=None, max_books: int | None = None):
    """Download and clean sci-fi novels from Gutenberg."""
    if config is None:
        config = Config()
    config.ensure_dirs()

    cache_dir = os.path.join(config.raw_dir, "gutenberg_cache")
    os.makedirs(cache_dir, exist_ok=True)
    limit = max_books or config.max_books

    # Step 1: Gather candidate sci-fi books
    print("Querying Gutendex API for science fiction novels...")
    api_books = query_gutendex_scifi()
    print(f"  Found {len(api_books)} books from API")

    # Combine seed list + API results, deduplicate by ID
    seen_ids = set()
    all_candidates = []

    for bid, title, author in SEED_SCIFI:
        if bid not in seen_ids:
            seen_ids.add(bid)
            all_candidates.append({"id": bid, "title": title, "authors": [author]})

    for book in api_books:
        if book["id"] not in seen_ids:
            seen_ids.add(book["id"])
            all_candidates.append(book)

    print(f"  Total unique candidates: {len(all_candidates)}")

    # Step 2: Download, clean, filter
    books = []
    for candidate in tqdm(all_candidates, desc="Downloading"):
        if len(books) >= limit:
            break

        text = download_book(candidate["id"], cache_dir)
        if text is None:
            continue

        cleaned = clean_gutenberg_text(text)
        word_count = count_words(cleaned)

        if word_count < config.min_book_words:
            continue

        author = candidate["authors"][0] if candidate.get("authors") else "Unknown"
        if isinstance(author, dict):
            author = author.get("name", "Unknown")

        books.append({
            "id": candidate["id"],
            "title": candidate["title"],
            "author": author,
            "word_count": word_count,
            "char_count": len(cleaned),
            "download_date": time.strftime("%Y-%m-%d"),
        })

        out_path = os.path.join(config.raw_dir, f"book_{candidate['id']}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        time.sleep(0.5)

    # Sort by Gutenberg ID for deterministic subset selection
    books.sort(key=lambda b: b["id"])

    metadata_path = os.path.join(config.raw_dir, "book_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(books, f, indent=2)

    total_words = sum(b["word_count"] for b in books)
    print(f"\nDownloaded {len(books)} science fiction novels")
    print(f"Total: {total_words:,} words")
    print(f"Saved metadata to {metadata_path}")
    return books


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Gutenberg sci-fi novels")
    parser.add_argument("--max", type=int, default=None, help="Max books to download")
    args = parser.parse_args()
    run(max_books=args.max)
