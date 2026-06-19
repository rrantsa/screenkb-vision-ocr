"""Database migration script — initialize screenkb SQLite schema.

Run once to create or upgrade the knowledge base:
    python migrate_db.py

Optional: specify a custom DB path:
    SCREENKB_DB=/path/to/screenkb.db python migrate_db.py
"""

import sys
from pathlib import Path

# Allow running from the tools/screenkb/ directory
sys.path.insert(0, str(Path(__file__).parent))

from screenkb.knowledge_base import init_db, get_db_path, get_all_rules


def main():
    db_path = get_db_path()
    print(f"Initializing database at: {db_path}")
    conn = init_db(db_path)

    rules = get_all_rules(conn)
    print(f"Schema ready. Default rules loaded: {len(rules)}")
    for r in rules:
        req = ", ".join(r["required_features"])
        print(f"  [{r['screen_type']:25s}] base={r['base_confidence']:.2f}  requires=[{req}]")

    conn.close()
    print("\nDone. Knowledge base is ready.")


if __name__ == "__main__":
    main()
