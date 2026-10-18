#!/usr/bin/env python3

# Read-only connection to the AFL stats database built by AFL-Data-Scraping, working against
# either SQLite or PostgreSQL.
#
# The database is chosen with the AFL_DB_URL environment variable, read from the environment or
# from the .env file next to this one:
#   sqlite:///afl_stats.db                      (default; relative path)
#   sqlite:////absolute/path/afl_stats.db
#   postgresql://user:password@host:5432/afl_stats
#
# PostgreSQL needs a driver: `pip install "psycopg[binary]"` (psycopg2 also works).
#
# SQL passed to Database.execute() uses "?" placeholders; they are translated to "%s"
# for PostgreSQL drivers.

import os
import sqlite3
import sys

DEFAULT_URL = "sqlite:///afl_stats.db"
ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def _load_env_file(path=ENV_FILE):
    """Set variables from a .env file (KEY=value or export KEY=value) that aren't already set.
    Values may refer to earlier variables as $VAR or ${VAR}."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export ") :]
            key, value = line.split("=", 1)
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            # $VAR / ${VAR} refer to earlier variables, as when the file is sourced in a shell
            os.environ.setdefault(key.strip(), os.path.expandvars(value))


class Database:
    def __init__(self, url):
        self.url = url
        if url.startswith("sqlite:///"):
            self.dialect = "sqlite"
            path = url[len("sqlite:///") :]
            self.connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        elif url.startswith(("postgresql://", "postgres://")):
            self.dialect = "postgresql"
            try:
                import psycopg

                self.connection = psycopg.connect(url)
                self.connection.read_only = True
            except ImportError:
                try:
                    import psycopg2
                except ImportError:
                    sys.exit('PostgreSQL needs a driver: pip install "psycopg[binary]"')
                self.connection = psycopg2.connect(url)
                self.connection.set_session(readonly=True)
        else:
            raise ValueError(
                f"Unsupported database URL: {url!r} (expected sqlite:/// or postgresql://)"
            )

    def _translate(self, sql):
        if self.dialect == "postgresql":
            return sql.replace("%", "%%").replace("?", "%s")
        return sql

    def execute(self, sql, params=()):
        if not isinstance(params, (tuple, list)):
            raise TypeError(
                f"params must be a tuple or list, not {type(params).__name__}"
            )
        cursor = self.connection.cursor()
        cursor.execute(self._translate(sql), tuple(params))
        return cursor

    def fetchone(self, sql, params=()):
        return self.execute(sql, params).fetchone()

    def fetchall(self, sql, params=()):
        return self.execute(sql, params).fetchall()

    def close(self):
        self.connection.close()


def createReaderConnection(url=None):
    _load_env_file()
    return Database(url or os.environ.get("AFL_DB_URL", DEFAULT_URL))
