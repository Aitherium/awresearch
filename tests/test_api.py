"""Tests for awresearch.api — Report, Claim, Source classes."""

import json

from awresearch.api import Claim, Report, Source


class TestSource:
    """Tests for Source class."""

    def test_source_creation(self):
        """Source can be created with URL and title."""
        source = Source(url="https://example.com", title="Example Page")
        assert source.url == "https://example.com"
        assert source.title == "Example Page"

    def test_source_with_metadata(self):
        """Source can store trust/freshness metadata."""
        source = Source(
            url="https://example.com",
            title="Example",
            domain="example.com",
            authority=0.85,
            freshness=0.92,
            trust=0.88,
        )
        assert source.authority == 0.85
        assert source.freshness == 0.92
        assert source.trust == 0.88

    def test_source_to_dict(self):
        """Source can be serialized to dict."""
        source = Source(
            url="https://example.com",
            title="Example",
            authority=0.85,
        )
        d = source.to_dict()
        assert d["url"] == "https://example.com"
        assert d["title"] == "Example"
        assert d["authority"] == 0.85


class TestClaim:
    """Tests for Claim class."""

    def test_claim_unsourced(self):
        """Claim with no sources is unsourced."""
        claim = Claim(text="This is a claim")
        assert not claim.is_sourced
        assert claim.sources == []

    def test_claim_sourced(self):
        """Claim with sources is sourced."""
        claim = Claim(text="This is a claim", sources=[1, 2])
        assert claim.is_sourced
        assert claim.sources == [1, 2]

    def test_claim_unsourced_with_reason(self):
        """Unsourced claim can have an explicit reason."""
        claim = Claim(
            text="This is general knowledge",
            sources=[],
            unsourced_reason="too general to cite",
        )
        assert not claim.is_sourced
        assert claim.unsourced_reason == "too general to cite"

    def test_claim_to_dict(self):
        """Claim can be serialized to dict."""
        claim = Claim(text="Test claim", sources=[1])
        d = claim.to_dict()
        assert d["text"] == "Test claim"
        assert d["sources"] == [1]
        assert d["is_sourced"] is True


class TestReport:
    """Tests for Report class."""

    def test_report_creation(self):
        """Report can be created with a question."""
        report = Report(question="What is X?")
        assert report.question == "What is X?"
        assert report.claims == []
        assert report.sources == []

    def test_report_with_sourced_claims(self):
        """Report can hold sourced claims and sources."""
        report = Report(question="What is X?")
        report.sources.append(
            Source(url="https://example.com", title="Example")
        )
        report.claims.append(Claim(text="X is Y", sources=[1]))

        assert len(report.sources) == 1
        assert len(report.claims) == 1
        assert report.claims[0].is_sourced

    def test_report_validate_clean(self):
        """Valid report with sourced claims validates cleanly."""
        report = Report(question="What is X?")
        report.sources.append(
            Source(url="https://example.com", title="Example")
        )
        report.claims.append(Claim(text="X is Y", sources=[1]))

        issues = report.validate()
        assert issues == []

    def test_report_validate_unsourced_claim_no_reason(self):
        """Report detects unsourced claim without reason."""
        report = Report(question="What is X?")
        report.claims.append(Claim(text="X is Y", sources=[]))

        issues = report.validate()
        assert len(issues) > 0
        assert "no source and no reason" in issues[0]

    def test_report_validate_unsourced_claim_with_reason(self):
        """Report accepts unsourced claim with explicit reason."""
        report = Report(question="What is X?")
        report.claims.append(
            Claim(
                text="Common knowledge",
                sources=[],
                unsourced_reason="too general to cite",
            )
        )

        issues = report.validate()
        assert issues == []

    def test_report_validate_broken_reference(self):
        """Report detects source reference out of bounds."""
        report = Report(question="What is X?")
        report.sources.append(
            Source(url="https://example.com", title="Example")
        )
        report.claims.append(Claim(text="X is Y", sources=[1, 5]))  # 5 doesn't exist

        issues = report.validate()
        assert len(issues) > 0
        assert "references source 5" in issues[0]

    def test_report_to_dict(self):
        """Report can be serialized to dict."""
        report = Report(question="What is X?")
        report.sources.append(
            Source(url="https://example.com", title="Example")
        )
        report.claims.append(Claim(text="X is Y", sources=[1]))

        d = report.to_dict()
        assert d["question"] == "What is X?"
        assert len(d["sources"]) == 1
        assert len(d["claims"]) == 1
        assert d["claims"][0]["is_sourced"] is True

    def test_report_to_dict_json_serializable(self):
        """Report dict can be converted to JSON."""
        report = Report(question="What is X?")
        report.sources.append(
            Source(
                url="https://example.com",
                title="Example",
                authority=0.85,
            )
        )
        report.claims.append(Claim(text="X is Y", sources=[1]))

        d = report.to_dict()
        json_str = json.dumps(d)  # Should not raise
        assert "What is X?" in json_str
        assert "https://example.com" in json_str

    def test_report_markdown_with_claims_and_sources(self):
        """Report renders as Markdown with citations."""
        report = Report(question="What is X?")
        report.sources.append(
            Source(url="https://example.com", title="Example Page")
        )
        report.sources.append(
            Source(url="https://other.com", title="Other Page")
        )
        report.claims.append(Claim(text="X is Y", sources=[1]))
        report.claims.append(Claim(text="X is also Z", sources=[1, 2]))

        markdown = report.markdown()
        assert "# What is X?" in markdown
        assert "X is Y [1]" in markdown
        assert "X is also Z [1], [2]" in markdown
        assert "https://example.com" in markdown
        assert "https://other.com" in markdown

    def test_report_markdown_with_unsourced_claim(self):
        """Report markdown includes unsourced claim explanation."""
        report = Report(question="What is X?")
        report.claims.append(
            Claim(
                text="General fact",
                sources=[],
                unsourced_reason="common knowledge",
            )
        )

        markdown = report.markdown()
        assert "General fact" in markdown
        assert "unsourced: common knowledge" in markdown

    def test_report_markdown_no_claims(self):
        """Report with no claims renders an empty note."""
        report = Report(question="What is X?")
        markdown = report.markdown()
        assert "No claims" in markdown


class TestReportIntegration:
    """Integration tests for Report with multiple claims/sources."""

    def test_multi_claim_multi_source_report(self):
        """Complex report with overlapping sources and claims."""
        report = Report(question="Compare solid-state batteries")
        report.sources.append(
            Source(
                url="https://arxiv.org/paper1",
                title="SSB Research 2024",
                domain="arxiv.org",
                authority=0.95,
                freshness=0.95,
            )
        )
        report.sources.append(
            Source(
                url="https://techcrunch.com/battery",
                title="Battery Startups",
                domain="techcrunch.com",
                authority=0.8,
                freshness=0.85,
            )
        )
        report.sources.append(
            Source(
                url="https://wikipedia.org/battery",
                title="Battery Wikipedia",
                domain="wikipedia.org",
            )
        )

        report.claims.append(
            Claim(text="Solid-state batteries use lithium metal anodes", sources=[1])
        )
        report.claims.append(
            Claim(
                text="Leading startups include Quantumscape and Samsung",
                sources=[2],
            )
        )
        report.claims.append(
            Claim(
                text="Energy density is 2-3x higher than lithium-ion",
                sources=[1, 2],
            )
        )
        report.claims.append(
            Claim(
                text="Cost reduction is the main barrier to adoption",
                unsourced_reason="consensus",
            )
        )

        # Validate
        issues = report.validate()
        assert issues == []

        # Serialize
        d = report.to_dict()
        assert len(d["sources"]) == 3
        assert len(d["claims"]) == 4

        # Markdown
        markdown = report.markdown()
        assert "[1]" in markdown
        assert "[2]" in markdown
        assert "Quantumscape" in markdown
