---
name: inspect-run
description: Read the full LLM message history (inputs, outputs, tool calls, errors, token usage, stop reasons) for an AIAutoProver run from its log folder. Use this skill ANY time the user asks something about a specific run that produced an autoProve log — e.g. "why did that run loop", "what error did the model see on retry 5", "how many tokens did it burn", "show me the prompt for the system-analysis call", "what did the code_explorer subagent for X return", "what was the LLM's last response before it died". The events.jsonl only contains metadata; the actual messages live in Postgres and this skill is the way to reach them. If the user asks about a run but does NOT provide a log path, proactively ask where the log is (e.g. ".certora_internal/autoProve/<timestamp>.events.jsonl" or the autoProve folder) before answering — do not guess or make claims about message content without inspecting it.
---

# inspect-run

## What this skill does

`events.jsonl` files in `.certora_internal/autoProve/` record start/end/checkpoint events with thread paths, but **not the actual LLM messages, tool-call arguments, or tool results**. Those live in Postgres (`langgraph_checkpoint_db`), keyed by the `thread_id`s that appear in the events `path` arrays.

This skill bundles a script that:

1. Takes any of: a `.events.jsonl` file, the matching `.log`, the `autoProve/` folder, or a project root.
2. Walks the events to discover every thread_id used in the run (both the top-level execution and any subagents).
3. Pulls the message list for any thread from Postgres via langgraph's `AsyncPostgresSaver`.
4. Prints a useful view (summary, filtered messages, or one full message).

## When the user hasn't given you a log path

If the user asks anything about a specific run but no log file/folder was mentioned in this conversation, **stop and ask** before answering. Don't speculate from memory or from code. A good ask:

> "Which run? Point me at the events.jsonl (or the `.certora_internal/autoProve/` folder, or the `.log` file) and I'll pull the messages."

If multiple runs sit in one folder, the script picks the newest and tells you which one. If the user names a specific timestamp, pass the full filename.

## How to run the script

The script imports the `composer` package, so it must run inside an environment where that import resolves (the AIAutoProver venv, or whatever Python the user has set up to work in this repo). Different users have different venv flows, so **do not assume a particular Python path**. The simplest first attempt:

```bash
python scripts/inspect_run.py <subcommand> <log-path> [flags]
```

(adjust the script path relative to your CWD). If this errors with `ModuleNotFoundError: composer` (or a similar import failure), **ask the user which Python / venv they want to use** for this repo and re-run with that interpreter. Don't guess venv paths.

### Subcommands

**Always start with `summary`** when you first look at a run. It tells you which thread_ids exist, how many messages each holds, and whether errors / max_tokens stops appeared. The output names every subagent thread so you can drill into them if the user's question is about one.

```bash
inspect_run.py summary <log-path>
```

Shows: top-level execution(s) with message count, type breakdown, error count, max_tokens count; then a list of subagent threads with parent annotations.

```bash
inspect_run.py messages <log-path> [--thread T] [--range A:B] [--errors-only] [--type AIMessage|ToolMessage|HumanMessage|SystemMessage] [--tool NAME] [--full]
```

Lists messages with one-line previews by default. `--thread` defaults to the thread_id of the run's top-level execution; pass an explicit thread_id (from `summary`) to inspect a subagent. Use `--errors-only` to find tool failures; `--tool cvl_document_ref` to find every invocation of of the tool `cvl_document_ref`; `--full` to expand the listed messages with complete content + tool-call args.

```bash
inspect_run.py message <log-path> <index> [--thread T]
```

Dumps one message at the given index in full, including content, response_metadata, usage_metadata, and tool_calls. Use this after `messages` narrows you to an interesting index.

## Workflow: how to answer different kinds of questions

The point of starting with `summary` is to anchor on real data before you reason. Don't paraphrase the user's question into an answer without seeing the messages.

- **"Why did the run fail / loop?"** → `summary` first. Look at error count and any `stop_reason=max_tokens` flags. Then `messages --errors-only` (or `messages --tool <name>`) to see the actual error text.
- **"What did the LLM submit on the Nth retry?"** → `summary`, then `messages --tool <toolname>`, then `message <index> --full` on the relevant one.
- **"What did the code_explorer for X return?"** → `summary` lists subagent thread_ids and their parent paths; pick the one whose description matches X (the events' `description` field is shown), then `messages --thread <subagent-thread-id>`.
- **"How expensive was this?"** → `messages` shows per-AIMessage token counts in the one-line view; sum them or look at the highest ones.

## Drilling into subagents

`summary` lists subagent thread_ids like `code_explorer-205f3e052d064476` along with their parent thread. To inspect one:

```bash
inspect_run.py messages <log-path> --thread code_explorer-205f3e052d064476
```

If the user's question is clearly about a specific subagent (e.g. "what did the OptimisticOracle code_explorer find?"), match the explorer by its `Code Explorer: <description>` start event — `summary` prints these so you can pick the right thread_id.

## What to do with the output

Read the script's output, identify the messages that actually answer the user's question, and quote the relevant content directly. Don't summarize a 200-line transcript into "looks like it worked" — pull specific message indices, error strings, or token counts and refer to them so the user can cross-check.

If the answer involves a long quote (e.g. a multi-paragraph tool-result), include enough of it that the user can see the evidence, and tell them how to re-run the command to see more.

## Failure modes

- **`ModuleNotFoundError: composer`** (or similar import error): you used the wrong Python. Ask the user which venv / Python interpreter they use for this repo and re-invoke with that one.
- **`connection refused` / DB errors**: the Postgres services aren't running. Tell the user — they need to start the composer DB stack before the script can fetch messages.
- **Empty message list / thread not found**: the run may have died before its first checkpoint, or the thread_id was mistyped. Re-run `summary` to confirm available threads.
