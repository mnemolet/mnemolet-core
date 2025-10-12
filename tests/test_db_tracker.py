from mnemolet_core.storage.db_tracker import (
    init_db,
    add_file,
    file_exists,
    list_files,
    mark_indexed,
    DB_PATH,
)
import os


def setup_module(module):
    """
    Remove db before tests (clean run).
    """
    if DB_PATH.exists():
        os.remove(DB_PATH)


def test_add_and_list_files():
    init_db()
    path = "example.txt"
    file_hash = "example_hash"

    add_file(path, file_hash)
    assert file_exists(file_hash) is True

    files = list_files()
    assert len(files) == 1
    assert files[0]["path"] == path

    mark_indexed(file_hash)
    indexed_files = list_files(indexed=True)
    assert len(indexed_files) == 1
