# Certora AI Composer

AI Composer is a tool for generating verified implementations from documentation and CVL specifications.

# Installation

## Requirements

You will need at least Python 3.11, Docker (with compose), and a Claude API key, and the ability to build
the documentation (see [here](https://github.com/Certora/Documentation/?tab=readme-ov-file#building-the-documentation)),
and a working, local installation of the prover (see [here](https://github.com/certora/certoraprover)). The Claude API Key should be in your
environment under `ANTHROPIC_API_KEY`.

## One-time DB setup

You will need to provision the various Postgres databases used by AI Composer. Do this as follows:
1. cd into `verisafe/scripts/`
2. run `docker compose create && docker compose start`. This will initalize a local postgres database. NB: no attempt has been made
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

## One-time prover setup

From the root of the Certora Prover repo, run `./gradlew copy-assets`. Ensure that your `CERTORA` environment
variable is configured to point to the output of this build (`CertoraProver/target`)

## AI Composer Requirements

Install the requirements for AI Composer via `pip3 install -r ./requirements.txt`. You may do this in
a virtual environment, and in such case you also need to install the dependencies for the `certora-cli`:
`pip install -r certora_cli_requirements.txt` from the `CertoraProver/scripts` folder, and optionally the Solidity compiler, if none is
available system-wide. Also be sure to activate this new virtual environment each time you want to run verisafe.

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

Where `cvl_input.spec` is the CVL specification the implementationmust conform to, `interface_file.sol`
contains an `interface` definition which the generated contract must implement, and `system_doc.txt`
is a text file containing a description of the overall system (defining key concepts, etc.)

AI Composer will iterate some number of times while it attempts to generate code. This process is *semi* automatic;
AI Composer may ask for help via the human in the loop tool, propose spec changes, or ask for requirement relaxation.
It is recommended that you "babysit" the process as it run.

A basic trace of what the tool is doing is displayed to stdout. You can enable `--debug` to see *very* verbose output, but
more friendly debugging options are described below.

Once generation is completed, the generated sources and the LLM commentary is dumped to stdout.

### Basic Options

A few options can help tweak your experience:

* `--prover-capture-output false` will have the prover runs invoked by the AI Composer print its output to stdout/stderr instead of being captured
* `--prover-keep-folders` will print the temporary directories used for the prover runs, and not clean them up
* `--debug-prompt-override PROMPT` will append whatever text you provide in `PROMPT` to the inital prompt. Useful for instructing the LLM to do different things
* `--tokens T` How many tokens to sample from the llm. This needs to be *relatively* high due to the amount of code that needs to be generated
* `--thinking-tokens T` how many tokens of the overall token budget should be used for thinking
* `--model` The name of the Anthropic model to use for the task. Defaults to sonnet
* `--thread-id` and `--checkpoint-id` are used for resuming workflows that crash or need tweaking (see below)
* `--summarization-threshold` enables the summarization of older messages after a certain threshold

### Resuming Workflows



## Development workflow

Whenever you change any of `graphcore`'s functions (which is imported from the $CERTORA folder) you will have to re-copy the assets from your EVMVerifier repo.

## Resuming a previously completed workflow with updated specification and files

Resuming can be done in one of two ways:

* using the materialize command of the `resume.py` to materialize the result of a prior run into a folder,
arbitrarily changing the contents of that folder, and then using the resume-dir command of `resume.py`, OR
* using resume-id with the thread-id and passing in an updated specification file

In either case, the generation workflow will be started with an already populated VFS, pulled either from the VFS result table (resume id),
or the materialized folder (resume dir). In addition, the generation workflow has been extended to save "resumption commentary",
which is a version of the "final commentary" produced by the tool, but intended for future resumptions.
Thus, when a workflow is resumed (either through resume-dir or resume-id), we pull in the "resume commentary" from the resumed run and seed the input prompt with this.
