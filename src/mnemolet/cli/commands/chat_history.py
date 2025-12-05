import click

from mnemolet.core.storage.chat_history import ChatHistory


@click.group("history")
def history():
    """Manage chat history."""
    pass


@history.command("list")
def list_history():
    """List all chat sessions."""
    h = ChatHistory()
    sessions = h.list_sessions()
    if not sessions:
        click.echo("No chat sessions found.")
        return
    for s in sessions:
        click.echo(f"{s['id']}: created at {s['created_at']}")
