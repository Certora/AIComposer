# VeriSafe POC

VeriSafe is a tool for generating verified implementations from documentation and CVL specifications.

# Installation

## Requirements

You will need at least Python 3.11, Docker, and a Claude API key, and the ability to build
the documentation (see [here](https://github.com/Certora/Documentation/?tab=readme-ov-file#building-the-documentation)),
and a working, local installation of the prover. The Claude API Key should be in your
environment under `ANTHROPIC_API_KEY`.

## One-time RAG setup

You will need to build the local RAG database used for CVL manual searches by the LLM.
Instructions are as follows:

1. cd into `verisafe/scripts/`
2. run `docker compose create && docker compose start`. This will initalize a local postgres database. NB: no attempt has been made
   to ensure this database is secure; caveat emptor
3. Run the base script `./gen_docs.sh`; if it completes without error, you should have `cvl_manual.html` in your directory
4. Create a new python virtual environment for the RAG build process by running `python3 -m venv somepath` where `somepath` is some path
   on your filesystem
5. Run `source somepath/bin/activate`.
6. Run `pip3 install -r ./rag_build_requirements.txt`
7. Run `python3 ./ragbuild.py cvl_manual.html`.
8. Run the command `deactivate`
9. (Optional) cleanup `somepath`

## One-time prover setup

From the root of the EVMVerifier repo, run `./gradlew copy-assets`. Ensure that your `CERTORA` environment
variable is configured to point to the output of this build (`EVMVerifier/target`)

## VeriSafe Requirements

Install the requirements for VeriSafe via `pip3 install -r ./requirements.txt`. You may do this in
a virtual environment, and in such case you also need to install the dependencies for the `certora-cli`:
`pip install -r certora_cli_requirements.txt` from the CertoraProver/scripts folder, and optionally the Solidity compiler, if none is
available system-wide. Also be sure to activate this new virtual environment each time you want to run verisafe.

# Usage

Once you have completed the above setup, you can run verisafe via:

```
python3 ./verisafe/main.py cvl_input.spec interface_file.sol system_doc.txt
```

Where `cvl_input.spec` is the CVL specification which VeriSafe attempts to conform to, `interface_file.sol`
contains an `interface` definition which the generated contract must implement, and `system_doc.txt`
is a text file containing a description of the overall system (defining key concepts, etc.)

VeriSafe will iterate some number of times until it either hits the recursion limit of langgraph, or it
generates code. A basic trace of what the tool is doing is printed to stdout. It may also ask for help
using the human in the loop tool.

A few options can help tweak your experience:

* `--prover-capture-output false` will have the prover runs invoked by the VeriSafe print their output to stdout/stderr instead of being captured
* `--prover-keep-folders` will print the temporary directories used for the prover runs, and not clean them up
* `--debug-prompt-override PROMPT` will append whatever text you provide in `PROMPT` to the inital prompt. Useful for instructing the LLM to do different things
* `--tokens T` How many tokens to sample from the llm. This needs to be *relatively* high due to the amount of code that needs to be generated
* `--thinking-tokens T` how many tokens of the overall token budget should be used for thinking
* `--model` The name of the Anthropic model to use for the task. Defaults to sonnet
* `--thread-id` and `--checkpoint-id` are used for resuming workflows that crash or need tweaking

## Development workflow

Whenever you change any of `graphcore`'s functions (which is imported from the $CERTORA folder) you will have to re-copy the assets from your EVMVerifier repo.
