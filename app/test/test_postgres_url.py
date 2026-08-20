"""
Tests for the Postgres DSN builder.

Credentials are URL components, so a password containing '@', ':', '/' or '?'
has to be percent-encoded or the URL parses wrongly. The real failure this
guards against, seen on the engine box:

    psycopg2.OperationalError: could not translate host name
    "!2026@localhost" to address

The '@' in the password ended the userinfo section early, so everything after
it was read as the hostname.

No database or engine dependencies required:
    python -m unittest discover -s app/test -p "test_*.py" -v
"""

import unittest
from urllib.parse import unquote, urlsplit


class _FakeSettings:
    """Stands in for Settings so the test does not need a .env or dotenv."""

    POSTGRES_HOST = "localhost"
    POSTGRES_PORT = 5432
    POSTGRES_DB = "xeliai_vectors"

    def __init__(self, user, password):
        self.POSTGRES_USER = user
        self.POSTGRES_PASSWORD = password

    @property
    def POSTGRES_URL(self):
        from app.core.config import Settings

        return Settings.POSTGRES_URL.fget(self)


class TestPostgresUrl(unittest.TestCase):
    def test_plain_credentials_round_trip(self):
        url = _FakeSettings("akili", "simplepass").POSTGRES_URL
        parts = urlsplit(url)
        self.assertEqual(parts.hostname, "localhost")
        self.assertEqual(parts.port, 5432)
        self.assertEqual(parts.username, "akili")
        self.assertEqual(parts.password, "simplepass")
        self.assertEqual(parts.path, "/xeliai_vectors")

    def test_password_with_at_sign_does_not_corrupt_the_host(self):
        # The production failure: an unescaped '@' made the parser read
        # "!2026@localhost" as the hostname.
        url = _FakeSettings("akili", "pa!2026@ss").POSTGRES_URL
        parts = urlsplit(url)
        self.assertEqual(parts.hostname, "localhost")
        # urlsplit returns raw components; the driver decodes them, so the
        # round-trip check has to decode too.
        self.assertEqual(unquote(parts.password), "pa!2026@ss")

    def test_password_with_colon_round_trips(self):
        url = _FakeSettings("akili", "we:ird").POSTGRES_URL
        self.assertEqual(unquote(urlsplit(url).password), "we:ird")

    def test_password_with_slash_and_question_mark_round_trips(self):
        url = _FakeSettings("akili", "a/b?c#d").POSTGRES_URL
        parts = urlsplit(url)
        self.assertEqual(unquote(parts.password), "a/b?c#d")
        self.assertEqual(parts.path, "/xeliai_vectors")

    def test_username_is_escaped_too(self):
        url = _FakeSettings("user@corp", "pw").POSTGRES_URL
        self.assertEqual(unquote(urlsplit(url).username), "user@corp")

    def test_percent_in_a_password_round_trips(self):
        # A literal '%' must be escaped as %25 or the decode is wrong.
        url = _FakeSettings("akili", "100%pure").POSTGRES_URL
        self.assertEqual(unquote(urlsplit(url).password), "100%pure")


if __name__ == "__main__":
    unittest.main()
