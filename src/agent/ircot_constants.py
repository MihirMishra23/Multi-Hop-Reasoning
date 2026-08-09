"""Dependency-free provenance constants shared by IRCoT launch code."""

OFFICIAL_IRCOT_COMMIT = "3c1820f698eea5eeddb4fba3c56b64c961e063e4"
OFFICIAL_IRCOT_URL = "https://github.com/StonyBrookNLP/ircot"
OFFICIAL_CORPUS_NAMES = {
    "hotpotqa": "hotpotqa",
    "2wiki": "2wikimultihopqa",
    "musique": "musique",
}


__all__ = ["OFFICIAL_CORPUS_NAMES", "OFFICIAL_IRCOT_COMMIT", "OFFICIAL_IRCOT_URL"]
