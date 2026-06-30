"""MCP prompt templates for org-roam assistant — registered on FastMCP instance."""

from mcp.server.fastmcp import FastMCP


def register_prompts(mcp: FastMCP) -> None:
    """Register all MCP prompts on the FastMCP instance."""

    @mcp.prompt(name='roam-assistant')
    def roam_assistant() -> str:
        """System prompt for a roam-aware research assistant."""
        return """You are a roam-aware research assistant with access to an org-roam personal knowledge graph.

You can:
- Search notes with `roam_search` to find relevant topics
- Get rich context with `roam_context` to understand how ideas connect
- Create notes with `roam_capture` to save new insights
- Dump research with `roam_research_dump` to save papers/findings from scite or web searches
- Append to notes with `roam_append` to add incremental findings
- Explore the graph with `roam_subgraph` and `roam_backlinks`
- Check recent activity with `roam_recent` and `roam_daily`

Guidelines:
1. When asked about a topic, first search for it with `roam_search`, then build context with `roam_context`.
2. When dumping research from scite or web, use `roam_research_dump` and link to existing topic notes.
3. When analyzing notes, use `roam_context` with depth=2 to see the full picture.
4. Always provide the node ID when referencing a specific note.
5. Tags help organize — suggest relevant tags when capturing new notes."""

    @mcp.prompt(name='research-note')
    def research_note(source: str) -> str:
        """Template for creating a structured research note from a source."""
        return f"""Create a structured research note from the following source.

Steps:
1. Extract key metadata: title, authors, abstract, DOI, URL, year, journal
2. Identify the main findings (3-5 bullet points)
3. Check if there's an existing topic note to link to using `roam_search`
4. Create the note using `roam_research_dump` with all extracted data
5. If a topic was found, include it as the `topic` parameter

Source to process:
{source}"""

    @mcp.prompt(name='analyze-notes')
    def analyze_notes(node_ids: str, question: str) -> str:
        """Analyze a set of linked notes and answer a question."""
        return f"""Analyze the following set of org-roam notes and answer the question.

First, use `roam_context` with depth=2 on the main nodes to understand connections.
Then synthesize an answer based on the content and relationships.

Nodes to analyze: {node_ids}
Question: {question}"""
