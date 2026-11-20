"""TensorBoard with loss groups first and empty scalar groups hidden."""

import io
import logging
import zipfile

from absl import app
from tensorboard import assets, main_lib, program


def patch_assets(original: bytes) -> bytes:
    # Tb bypasses its scalar tag filter when no runs are selected. require a selection
    replacements = {
        b"!n||!o||r.plugin!==_n.SCALARS||i.has(r.tag)": b"r.plugin!==_n.SCALARS||(o&&(!n||i.has(r.tag)))",
        # Prioritize loss, including the previous names still present in old logs.
        b"t.slice().sort((i,o)=>Qy(i.tag,o.tag))": (
            b't.slice().sort((i,o)=>{const rank=t=>t.startsWith("loss/")?0:'
            b'/^(0 )?Overall Loss\\//.test(t)?1:2;return rank(i.tag)-rank(o.tag)||Qy(i.tag,o.tag)})'
        ),
    }
    with zipfile.ZipFile(io.BytesIO(original)) as source:
        script = source.read("index.js")
        if any(script.count(before) != 1 for before in replacements):
            logging.warning("TensorBoard assets changed; loss-group customizations were not applied.")
            return original
        for before, after in replacements.items():
            script = script.replace(before, after)

        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as target:
            for entry in source.infolist():
                content = source.read(entry)
                if entry.filename == "index.js":
                    content = script
                elif entry.filename == "index.html":
                    # Do not reuse the browser's cached, unmodified JavaScript.
                    content = content.replace(b"index.js?_file_hash=", b"index.js?_file_hash=onetrainer-v2-")
                target.writestr(entry, content)
        return output.getvalue()


def assets_zip_provider():
    with assets.get_default_assets_zip_provider()() as source:
        return io.BytesIO(patch_assets(source.read()))


if __name__ == "__main__":
    main_lib.global_init()
    tensorboard = program.TensorBoard(assets_zip_provider=assets_zip_provider)
    app.run(tensorboard.main, flags_parser=tensorboard.configure)
