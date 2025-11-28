import functools
import re
from multiprocessing import Pool
from pathlib import Path

from modules.ui.models.SingletonConfigModel import SingletonConfigModel
from modules.util.enum.BulkEditMode import BulkEditMode


def _edit_text(config, read_only, file):
    try:
        changed = False
        with open(file, "r+", encoding="utf-8") as f:
            content = original_content = f.read()
            if config["add_text"] != "":
                content = f"{config['add_text']} {content}" if config["add_mode"] == BulkEditMode.PREPEND else f"{content} {config['add_text']}"
            if config["remove_text"] != "":
                content = content.replace(config["remove_text"], "")
            if config["replace_text"] != "" and config["replace_with"] != "":
                content = content.replace(config["replace_text"], config["replace_with"])
            if config["regex_pattern"] != "" and config["regex_replace"] != "":
                regex = re.compile(config["regex_pattern"])
                content = regex.sub(config["regex_replace"], content)

            changed = content != original_content

            if not read_only and changed:
                f.seek(0)
                f.truncate()
                f.write(content)
    except (OSError, re.error):
        return content, False

    return content, changed


class BulkModel(SingletonConfigModel):
    def __init__(self):
        super().__init__({
            "directory": "",
            "add_text": "",
            "add_mode": BulkEditMode.PREPEND,
            "remove_text": "",
            "replace_text": "",
            "replace_with": "",
            "regex_pattern": "",
            "regex_replace": "",
        })

        self.pool = None

    def terminate_pool(self):
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
            self.pool = None


    def bulk_edit(self, read_only=False, preview_n=None, progress_fn=None):
        base_path = Path(self.get_state("directory"))
        files = list(base_path.glob("*.txt"))

        if self.pool is None:
            self.pool = Pool()

        if preview_n is not None and read_only:
            files = files[:preview_n]

        with self.critical_region_read():
            result = self.pool.map(functools.partial(_edit_text, self.config, read_only), files)

        total = len(result)
        processed = len([r for r in result if r[1]])
        skipped = total - processed

        if preview_n is not None:
            result = result[:preview_n]

        if progress_fn is not None:
            if read_only:
                progress_fn({"status": f"Previewing {total} files: {processed} will be processed, {skipped} will be skipped", "data": "\n\n".join([r[0] for r in result])})
            else:
                progress_fn({"status": f"Edited {total} files: {processed} processed, {skipped} skipped", "data": "\n\n".join([r[0] for r in result])})
