"""Autoprove run report.

A final, best-effort phase of the source autoprove pipeline that turns the
per-component property dumps (``certora/properties/*.json``) and the prover
verdicts (fetched per component run via ProverOutputUtility) into a single
audit-style report: every inferred property and CVL rule, each rule's prover
verdict, and an LLM-grouped set of high-level "P-NN" properties. Written to
``certora/ap_report/report.json`` (HTML is opt-in via ``render``).
"""
