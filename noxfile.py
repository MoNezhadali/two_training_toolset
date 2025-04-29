"""Noxfile to run automation tests."""

# Standard library imports
import os

# Third party imports
import nox

# BACKEND_DIR = "backend"
CI_SCRIPTS_DIR = ".github/scripts"

# Stop Nox if python interpreter was not found
nox.options.error_on_missing_interpreters = True


@nox.session
def tests(session):
    """Run tests for both sklearn versions."""
    # Install nox requirements
    session.install("-r", "requirements.txt")
    session.install("-r", "validation_requirements.txt")
    # Run tests
    session.run("bash", "tests" + os.sep + "unit_tests", external=True)


@nox.session
def formatting(session):
    """Run formatting checks in current interpreter for now."""
    # Install nox requirements
    session.install("-r", "requirements.txt")
    session.install("-r", "validation_requirements.txt")
    # Run formatting checks
    session.run("bash", f"{CI_SCRIPTS_DIR}/check_formatting.sh", external=True)


@nox.session
def linting(session):
    """Run linter checks in current interpreter for now."""
    # Install nox requirements
    session.install("-r", "requirements.txt")
    session.install("-r", "validation_requirements.txt")
    # Run linter
    session.run("bash", f"{CI_SCRIPTS_DIR}/linting.sh", external=True)


@nox.session
def security(session):
    """Run security checks in current interpreter for now."""
    # Install nox requirements
    session.install("-r", "requirements.txt")
    session.install("-r", "validation_requirements.txt")
    session.install("bandit[toml]")
    session.install("pip-audit")
    # Run security checks
    session.run("bash", f"{CI_SCRIPTS_DIR}/security.sh", external=True)
