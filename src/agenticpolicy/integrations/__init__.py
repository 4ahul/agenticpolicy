"""Framework adapters.

Each submodule imports its framework lazily and raises
:class:`~agenticpolicy.exceptions.IntegrationNotInstalled` with the right pip
command if it is missing, so a user who only wants LangChain never needs
LlamaIndex installed.
"""

__all__ = ["langchain_", "llamaindex_", "langgraph_"]
