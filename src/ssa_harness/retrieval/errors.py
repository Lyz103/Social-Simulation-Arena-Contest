"""Explicit errors at the boundary with the surrounding harness."""


class ResearchError(Exception):
    """A research bundle could not be produced safely."""


class ValidationError(ResearchError):
    pass


class TemporalPolicyError(ResearchError):
    pass


class ProviderError(ResearchError):
    pass


class MissingAPIKey(ProviderError):
    pass


class BudgetExceeded(ResearchError):
    pass


class EmptyCorpus(ResearchError):
    pass


class ArchiveConflict(ResearchError):
    pass


class ArchiveCorruption(ResearchError):
    pass


class WorkInProgress(ResearchError):
    pass
