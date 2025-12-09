"""Keyword-based search pipeline."""

import re

from src.data import CPRARequest, Email

from .base import SearchPipeline, SearchResult


class KeywordSearchPipeline(SearchPipeline):
    """Search pipeline using keyword matching."""

    def __init__(
        self,
        match_mode: str = "any",
        case_sensitive: bool = False,
        apply_exclusions: bool = True,
        use_secondary_keywords: bool = True,
    ):
        """Initialize keyword search pipeline.

        Args:
            match_mode: 'any' (OR) or 'all' (AND) for keyword matching
            case_sensitive: Whether to do case-sensitive matching
            apply_exclusions: Whether to exclude docs with exclude_keywords
            use_secondary_keywords: Whether to include secondary keywords
        """
        self.match_mode = match_mode
        self.case_sensitive = case_sensitive
        self.apply_exclusions = apply_exclusions
        self.use_secondary_keywords = use_secondary_keywords

    @property
    def name(self) -> str:
        return "Keyword Search"

    def _normalize(self, text: str) -> str:
        """Normalize text for matching."""
        if not self.case_sensitive:
            return text.lower()
        return text

    def _keyword_matches(self, keyword: str, text: str) -> bool:
        """Check if keyword appears in text.

        Uses word boundary matching to avoid partial matches
        (e.g., "lead" shouldn't match "leader").
        """
        normalized_keyword = self._normalize(keyword)
        normalized_text = self._normalize(text)

        # Use word boundaries for matching
        pattern = r"\b" + re.escape(normalized_keyword) + r"\b"
        return bool(re.search(pattern, normalized_text))

    def _get_keywords(self, request: CPRARequest) -> list[str]:
        """Get keywords to search for from request."""
        keywords = list(request.primary_keywords)
        if self.use_secondary_keywords:
            keywords.extend(request.secondary_keywords)
        return keywords

    def search(
        self, request: CPRARequest, emails: list[Email]
    ) -> list[SearchResult]:
        """Search for emails matching request keywords.

        Args:
            request: CPRA request with keywords
            emails: Emails to search

        Returns:
            List of SearchResult for matching emails (score=1.0 for match)
        """
        keywords = self._get_keywords(request)
        exclude_keywords = request.exclude_keywords if self.apply_exclusions else []

        results = []
        for email in emails:
            text = email.text

            # Check exclusions first
            if exclude_keywords:
                excluded = any(
                    self._keyword_matches(kw, text) for kw in exclude_keywords
                )
                if excluded:
                    continue

            # Find matching keywords
            matched = [kw for kw in keywords if self._keyword_matches(kw, text)]

            # Determine if this is a match based on mode
            if self.match_mode == "any":
                is_match = len(matched) > 0
            else:  # "all"
                is_match = len(matched) == len(keywords)

            if is_match:
                results.append(
                    SearchResult(
                        email_id=email.id,
                        score=1.0,
                        matched_terms=matched,
                    )
                )

        # Sort by number of matched terms (more matches = better)
        results.sort(key=lambda r: len(r.matched_terms), reverse=True)
        return results
