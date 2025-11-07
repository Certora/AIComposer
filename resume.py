import pathlib
import psycopg
from typing import cast
import verisafe.certora as _

from verisafe.input.parsing import resume_workflow_parser
from verisafe.workflow.factories import create_llm
from verisafe.workflow.executor import execute_cryptosafe_workflow
from verisafe.input.types import ResumeIdData, NativeFS, ResumeFSData
from verisafe.audit.db import AuditDB

def main() -> int:
    """Main entry point for the CryptoSafe tool."""
    parser = resume_workflow_parser()
    args = parser.parse_args()

    input_data: ResumeIdData | ResumeFSData

    match args.command:
        case "materialize":
            conn = psycopg.connect(args.audit_db)
            audit = AuditDB(conn)
            res = audit.get_resume_artifact(args.src_thread_id)
            out_dir = pathlib.Path(args.target)
            out_dir.mkdir(exist_ok=True, parents=True)

            if not out_dir.is_dir():
                raise RuntimeError(f"output dir {args.target} is not a directory")
            session_id_file = out_dir / ".session-id"
            if session_id_file.is_file() and \
                (curr_id := session_id_file.read_text().strip()) != args.src_thread_id:
                print(f"Refusing to materialize in a folder that appears to be a materialization of {curr_id}")
                print("You can remove the .session-id file to override this behavior")
                return 1
            for (p, cont) in res.vfs:
                output_path = out_dir / p
                output_path.parent.mkdir(exist_ok=True, parents=True)
                output_path.write_bytes(cont)
            session_id_file.write_text(args.src_thread_id)
            return 0
        case "resume-dir" | "resume-id":
            commentary: str | None = None
            if args.commentary is not None:
                if args.commentary.startswith('@'):
                    commentary = pathlib.Path(args.commentary[1:]).read_text()
                else:
                    commentary = args.commentary
            new_system=NativeFS(pathlib.Path(args.updated_system)) if args.updated_system is not None else None
            if args.command == "resume-dir":
                input_data = ResumeFSData(
                    comments=commentary,
                    new_system=new_system,
                    file_path=args.working_dir,
                    thread_id=args.src_thread_id
                )
            else:
                assert args.command == "resume-id"
                input_data = ResumeIdData(
                    thread_id=args.src_thread_id,
                    new_spec=NativeFS(pathlib.Path(args.new_spec)),
                    comments=commentary,
                    new_system=new_system
                )

    llm = create_llm(args)

    print("Starting VeriSafe resumption workflow...")
    return execute_cryptosafe_workflow(
        llm=llm,
        input=input_data,
        workflow_options=args
    )

if __name__ == "__main__":
    import sys
    sys.exit(main())
