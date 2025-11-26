import click


@click.command()
@click.option(
    "--host", default="127.0.0.1", show_default=True, help="Host to bind server to."
)
@click.option(
    "--port",
    default=8000,
    type=int,
    show_default=True,
    help="Port the server should run on.",
)
@click.option("--reload", is_flag=True, help="Enable auto-reload (for dev).")
def serve(host: str, port: int, reload: bool):
    """
    Start MnemoLet API server.
    """
    import uvicorn

    click.echo(f"Staring API server on http://{host}:{port} (reload={reload})")

    uvicorn.run(
        "mnemolet_core.app:app",
        host=host,
        port=port,
        reload=reload,
        log_config=None,
    )
