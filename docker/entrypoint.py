"""Contest entrypoint.

Kept in the exact location the contest contract expects
(``ENTRYPOINT ["python", "/app/docker/entrypoint.py"]``); the implementation
lives in the package so it can be imported and tested without Docker.
"""

import sys
from pathlib import Path

# ``python /app/docker/entrypoint.py`` puts /app/docker on sys.path.  The image
# sets PYTHONPATH=/app, but keeping this local makes the file runnable from a
# checkout as well.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greenchallenge.entrypoint import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
