import tempfile
from pathlib import Path
from mnemolet_core.ingestion.txt_loader import load_txt_files
from mnemolet_core.ingestion.utils import hash_file

def test_load_txt_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # create tmp files
        (tmp_path / "file1.txt").write_text("Hello world", encoding="utf-8")
        (tmp_path / "file2.txt").write_text("Another file", encoding="utf-8")
        (tmp_path / "empty.txt").write_text("", encoding="utf-8")

        files = load_txt_files(tmpdir)

        # skip empty files
        assert len(files) == 2

        contents = [f["content"] for f in files]
        assert "Hello world" in contents
        assert "Another file" in contents

        # check hash
        for f in files:
            assert f["hash"] == hash_file(Path(f["path"]))
