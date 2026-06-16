"""Errors for helper functions."""


class NoAtomRecordsError(ValueError):
    """No ATOM records means no molcules to import."""

    def __init__(self, pdb_path: str) -> None:
        super().__init__(
            f"No ATOM records found in {pdb_path}",
        )


class InvalidAAtypesError(ValueError):
    """Invalid aatypes."""


class TooManyChainsError(ValueError):
    """PDB format supports maximum of 62 chains."""

    def __init__(self, pdb_max_chains: str) -> None:
        super().__init__(
            f"The PDB format supports at most {pdb_max_chains} chains.",
        )
