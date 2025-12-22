from typing import Dict, Callable, Union
import os
import zlib
import anthropic
import pathlib
from io import BytesIO

from composer.input.types import CommandLineArgs, InputData, UploadedFile, FileSource, NativeFS, InMemorySource

def upload_file_if_needed(client: anthropic.Anthropic, source: FileSource, uploaded_files: Dict[str, str], log: Callable[[str], None] = print) -> UploadedFile:
    """Upload a file if not already uploaded, return UploadedFile."""
    content = source.bytes_contents
    crc_hex = hex(zlib.crc32(content))
    basename = source.basename
    crc_basename = f"{crc_hex}_{basename}"
    
    if crc_basename not in uploaded_files:
        log(f"Uploading {basename}... (canonical name {crc_basename})")
        uploaded_file = client.beta.files.upload(
            file=(crc_basename, BytesIO(content), "text/plain")
        )
        log(f"Uploaded {basename} with ID: {uploaded_file.id}")
        return UploadedFile(
            file_id=uploaded_file.id, 
            basename=basename, 
            path=getattr(source, 'path', basename),
            _content=source.string_contents
        )
    else:
        log(f"Found existing {basename} with ID: {uploaded_files[crc_basename]} (canonical name {crc_basename})")
        return UploadedFile(
            file_id=uploaded_files[crc_basename], 
            basename=basename, 
            path=getattr(source, 'path', basename),
            _content=source.string_contents
        )

def resolve_file_source(val: Union[str, dict, FileSource]) -> FileSource:
    """Resolve a path string or a JSON object into a FileSource."""
    if isinstance(val, str):
        return NativeFS(pathlib.Path(val))
    if isinstance(val, dict):
        # Expecting {"name": "...", "content": "..."}
        return InMemorySource(val["name"], val["content"])
    return val

def upload_input(i: CommandLineArgs, log: Callable[[str], None] = print) -> InputData:
    client = anthropic.Anthropic()
    d: Dict[str, str] = {}
    for f in client.beta.files.list():
        d[f.filename] = f.id

    # Upload the three input files, resolving them first
    interface_file = upload_file_if_needed(client, resolve_file_source(i.interface_file), d, log)
    spec_file = upload_file_if_needed(client, resolve_file_source(i.spec_file), d, log)
    system_doc_file = upload_file_if_needed(client, resolve_file_source(i.system_doc), d, log)

    return InputData(
        spec=spec_file,
        system_doc=system_doc_file,
        intf=interface_file,
    )
