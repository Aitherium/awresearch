"""Test configuration and fixtures."""

import pytest


@pytest.fixture
def fake_search_results():
    """Fake search results for testing."""
    return {
        "query": "test query",
        "results": [
            {
                "title": "Research on Test Topic",
                "url": "https://example.com/research",
                "snippet": "This is a test research article about the topic.",
            },
            {
                "title": "Another Study",
                "url": "https://example.org/study",
                "snippet": "Another important study on the same subject.",
            },
        ],
    }
