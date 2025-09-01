"""
Database access for the two databases (Siege and AI)
"""

import os
import datetime
import sqlite3
from enum import Enum
from dotenv import load_dotenv
from deps.log import print_error_log

load_dotenv()

EVENT_CONNECT = "connect"
EVENT_DISCONNECT = "disconnect"


class DBName(str, Enum):
    """
    Each database name.
    Must match the environment variable that start with the name followed with _DATABASE_PATH
    """

    SIEGE = "SIEGE"
    AI = "AI"


def adapt_datetime(dt):
    """Convert a datetime object to a string"""
    return dt.isoformat()


def convert_datetime(s):
    """Convert a string to a datetime object"""
    return datetime.datetime.fromisoformat(s)


class DatabaseManager:
    """
    Connect to one of the databases

    Example:
    with DatabaseManager.get_database_manager() as db:
        cursor = db.get_cursor(DBName.SIEGE)
        cursor.execute("SQL HERE")
        results = cursor.fetchall()
        print(results)
    """

    def __init__(self):
        sqlite3.register_adapter(datetime.datetime, adapt_datetime)
        sqlite3.register_converter("datetime", convert_datetime)

        self._databases = {}
        for key in ("SIEGE", "AI"):
            path = os.getenv(f"{key}_DATABASE_PATH")
            if not path:
                raise ValueError(f"{key}_DATABASE_PATH is not set")
            conn = sqlite3.connect(
                path, check_same_thread=False, detect_types=sqlite3.PARSE_DECLTYPES
            )
            self._databases[key] = {"conn": conn, "cursor": conn.cursor()}

    @classmethod
    def get_database_manager(cls) -> "DatabaseManager":
        """
        Create a single instance of DatabaseManager
        """
        return cls()

    def get_conn(self, name: DBName) -> sqlite3.Connection:
        """
        Get a connection access to a specific database
        """
        return self._databases[name]["conn"]

    def get_cursor(self, name: DBName) -> sqlite3.Cursor:
        """
        Get a cursor for a specific database
        """
        return self._databases[name]["cursor"]

    def close(self):
        """
        Close the connection for all databases
        """
        for key, db in self._databases.items():
            try:
                db["cursor"].close()
            except Exception as e:
                print_error_log(f"DatabaseManager.close({key} cursor): {e}")
            finally:
                db["cursor"] = None

            try:
                db["conn"].close()
            except Exception as e:
                print_error_log(f"DatabaseManager.close({key} conn): {e}")
            finally:
                db["conn"] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
