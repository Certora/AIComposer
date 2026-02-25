# Certora AI Composer

AI Composer is a tool for generating verified implementations from documentation and CVL specifications.

# Installation

## Requirements

You will need at least Python 3.12, Docker (with compose), and a Claude API key, and the ability to build
the documentation (see [here](https://github.com/Certora/Documentation/?tab=readme-ov-file#building-the-documentation)),
and a working, local installation of the prover (see [here](https://github.com/certora/certoraprover)). The Claude API Key should be in your
environment under `ANTHROPIC_API_KEY`.

## One-time DB setup

You will need to provision the various Postgres databases used by AI Composer. Do this as follows:

1. cd into `scripts/`
2. run `docker compose create && docker compose start`. This will initialize a local postgres database. NB: no attempt has been made
   to ensure this database is secure; caveat emptor

NB: You will need to restart this docker image each time your host computer restarts, unless you adjust the restart policy.

## One-time RAG Setup

You will need to build the local RAG database used for CVL manual searches by the LLM.
Instructions are as follows:

1. Run the base script `./gen_docs.sh`; if it completes without error, you should have `cvl_manual.html` in your directory
2. Create a new python virtual environment for the RAG build process by running `python3 -m venv somepath` where `somepath` is some path
   on your filesystem
3. Run `source somepath/bin/activate`.
4. Run `pip3 install -r ./rag_build_requirements.txt`
5. Run `python3 ./ragbuild.py cvl_manual.html`.
6. Run the command `deactivate`
7. (Optional) cleanup `somepath`

## One-time Extended RAG Setup (for Vacuity Analyzer)

The vacuity analyzer requires additional prover documentation beyond the CVL manual. If you plan to use the vacuity analyzer, follow these additional steps:

1. Run the extended documentation build script: `./gen_docs_extended.sh`; if it completes without error, you should have `extended_manual.html` in your directory
2. Using the same virtual environment from the previous RAG setup (or create a new one):
   - Run `source somepath/bin/activate`
   - Run `pip3 install -r ./rag_build_requirements.txt` (if not already installed)
   - Run `python3 ./ragbuild.py extended_manual.html --connection "postgresql://extended_rag_user:rag_password@localhost:5432/extended_rag_db"`
   - Run `deactivate`

**Note:** The cex-analyzer and AI Composer use the standard `rag_db` (CVL-only), while vacuity-analyzer defaults to `extended_rag_db` (CVL + prover docs). You can override this with the `--rag-db` flag if needed.

## One-time prover setup

From the root of the Certora Prover repo, run `./gradlew copy-assets`. Ensure that your `CERTORA` environment
variable is configured to point to the output of this build (`CertoraProver/target`)

## AI Composer Requirements

Install the requirements for AI Composer via `pip3 install -r ./requirements.txt`. You may do this in
a virtual environment, and in such case you also need to install the dependencies for the `certora-cli`:
`pip install -r certora_cli_requirements.txt` from the `CertoraProver/scripts` folder, and optionally the Solidity compiler, if none is
available system-wide. Also be sure to activate this new virtual environment each time you want to run AI Composer.

## Solidity Compilers

AI Composer assumes that the solidity compiler is available on your `$PATH` and follows the naming convention `solcX.Y`, where `X` and `Y`
are taken from the Solidity version numbers: `0.X.Y`. For example, to make solc version 0.8.29 available to AI Composer, you must ensure
that an executable `solc8.29` is somewhere on your path. Currently the LLM is prompted to use solidity version 0.8.29 but you can adjust
the prompts as needed.

# Usage

AI Composer is primarily a command line tool, with some more graphical debugging utilities available for use.

## Basic Operation

Once you have completed the above setup, you can run AI Composer via:

```
python3 ./main.py cvl_input.spec interface_file.sol system_doc.txt
```

Where `cvl_input.spec` is the CVL specification the implementation must conform to, `interface_file.sol`
contains an `interface` definition which the generated contract must implement, and `system_doc.txt`
is a text file containing a description of the overall system (defining key concepts, etc.)

AI Composer will iterate some number of times while it attempts to generate code. This process is _semi_ automatic;
AI Composer may ask for help via the human in the loop tool, propose spec changes, or ask for requirement relaxation.
It is recommended that you "babysit" the process as it runs.

A basic trace of what the tool is doing is displayed to stdout. You can enable `--debug` to see _very_ verbose output, but
more friendly debugging options are described below.

Once generation is completed, the generated sources and the LLM commentary is dumped to stdout.

### Basic Options

A few options can help tweak your experience:

- `--prover-capture-output false` will have the prover runs invoked by the AI Composer print its output to stdout/stderr instead of being captured
- `--prover-keep-folders` will print the temporary directories used for the prover runs, and not clean them up
- `--debug-prompt-override PROMPT` will append whatever text you provide in `PROMPT` to the initial prompt. Useful for instructing the LLM to do different things
- `--tokens T` How many tokens to sample from the LLM. This needs to be _relatively_ high due to the amount of code that needs to be generated
- `--thinking-tokens T` how many tokens of the overall token budget should be used for thinking
- `--model` The name of the Anthropic model to use for the task. Defaults to sonnet
- `--thread-id` and `--checkpoint-id` are used for resuming workflows that crash or need tweaking (see below)
- `--summarization-threshold` enables the summarization of older messages after a certain threshold

### Resuming Workflows

The `--thread-id` and `--checkpoint-id` options allow you to resume AI Composer execution from a specific point in time. Together, these identifiers describe a checkpoint in the execution history where the workflow can be resumed.

**Thread ID**: Identifies a specific execution session of AI Composer. This is displayed early in the output when starting a workflow:

```
Selected thread id: crypto_session_6511ace2-cfbf-11f0-aeb6-e8cf83d12a2d
```

**Checkpoint ID**: Identifies a specific point within that session. This is displayed throughout execution as the workflow progresses:

```
current checkpoint: 1f0cfbf9-bbd9-6365-8001-90d0fca3dbdf
```

To resume from a specific checkpoint, provide both identifiers:

```
python3 ./main.py --thread-id crypto_session_6511ace2-cfbf-11f0-aeb6-e8cf83d12a2d --checkpoint-id 1f0cfbf9-bbd9-6365-8001-90d0fca3dbdf cvl_input.spec interface_file.sol system_doc.txt
```

This will restart execution from exactly that point in the workflow. NB the checkpoint ID does _not_ need to be the most recent; you can "time travel" if you decide
you dislike a decision you made previously.

## Debugging Options

### Debug Console

During execution, you can pause the current workflow by sending SIGINT (usually by hitting Ctrl+C). Once the workflow reaches a
point of quiescence, you will be dropped into the "Debug Console". This console allows you to explore the current state of the implementation,
and review the entire message history. You can also use this console to provide explicit guidance; this guidance is echoed to the LLM verbatim.

The message history does NOT preserve messages across summarization boundaries.

### Trace Visualizer

After completion of a session, if you wish to see a visualization of the entire process you can use the `traceDump.py` script.

The basic usage is:

```
python3 scripts/traceDump.py thread-id conn-string out-file
```

Where `thread-id` is the thread ID for the session you wish to visualize. `conn-string` is the PostgreSQL string for connecting to the audit database, this should be
`postgresql://audit_db_user:audit_db_password@localhost:5432/audit_db` unless you have changed where audit data is stored. `out-file` is the name of an HTML file into
which the visual will be dumped.

### Exporting the Output

To get the final deliverable from AI Composer, use the VFS materializer like so:

```
python3 ./resume.py materialize thread-id path
```

where `thread-id` is the thread ID of the session whose output you wish to view, and `path` the path to a directory into which the resulting VFS is dumped.

## Meta-Iteration

Once AI Composer finishes generation, you can refine/adjust the specification and resume generation, seeding the process
with the output of a prior session. This is referred to as "meta-iteration".

Meta iteration can be done in one of two ways:

- use `materialize` command of `resume.py` (described above) to materialize the result of a prior run into a folder,
  arbitrarily changing the contents of that folder, and then using the `resume-dir` command of `resume.py`, OR
- using `resume-id` with the thread ID of a completed run and passing in an updated specification file

In the former case, the invocation looks like this:

```
python3 resume.py resume-dir thread-id path
```

Here `thread-id` is the thread ID of the workflow whose contents were materialized into `path`, the directory containing the changed
project files.

In the latter case, the invocation is:

```
python3 resume.py resume-id thread-id new-spec
```

where `thread-id` is the thread ID of the workflow on which you want to iterate, and `new-spec` is the path
to the updated/refined spec file to use for the next iteration.

# Disclaimer

AI Composer is a research prototype released by Certora Labs. The code generated by AI Composer should **not** be
placed into production without thorough vetting/testing/auditing.
