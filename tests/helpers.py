from pathlib import Path
from queue import Queue

from PIL import Image


def create_dummy_image(
    path: Path,
    width: int = 10,
    height: int = 10,
    mode: str = "RGB",
    format: str = "PNG",
):
    """Creates a dummy image file at the specified path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new(mode, (width, height), color="blue")
    img.save(path, format=format)
    return path


def create_dummy_text_file(path: Path, content: str = "caption"):
    """Creates a dummy text file with the given content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
    return path


def setup_test_directory(tmp_path, dirname, file_specs):
    """Create a test directory with specified files."""
    d = tmp_path / dirname
    d.mkdir()
    created_files = []

    for spec in file_specs:
        filename, is_image = spec[0], spec[1]
        kwargs = spec[2] if len(spec) > 2 else {}

        if is_image:
            created_files.append(create_dummy_image(d / filename, **kwargs))
        else:
            content = kwargs.get('content', "caption")
            created_files.append(create_dummy_text_file(d / filename, content))

    return d, created_files


def assert_message_in_queue(queue: Queue, message_type: str, message_content: str):
    """Check if a specific message exists in the queue without consuming it."""
    messages = []
    found = False

    while not queue.empty():
        msg = queue.get_nowait()
        messages.append(msg)
        if msg[0] == message_type and message_content in msg[1]:
            found = True

    # Restore messages to queue
    for msg in messages:
        queue.put_nowait(msg)

    return found
