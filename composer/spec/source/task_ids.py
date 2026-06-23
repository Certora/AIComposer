"""Canonical ``run_task`` task_ids for the AutoProve pipeline phases.

These strings are a contract between the orchestration that sets them on
``TaskInfo`` (``pipeline.py`` / ``common_pipeline.py``) and anything that keys
off them — notably the fake-LLM tape lanes in
``composer/testing/ui_harness_autoprove_Counter.py``, which route scripted responses by
task_id. Keeping the strings (and the per-component format) here means a rename
is a single edit instead of a silent desync that only fails at smoke-run time.
"""

SYSTEM_ANALYSIS_TASK_ID = "system-analysis"
HARNESS_TASK_ID = "harness"
AUTOSETUP_TASK_ID = "autosetup"
SUMMARIES_TASK_ID = "summaries"
INVARIANTS_TASK_ID = "invariants"
INVARIANT_CVL_TASK_ID = "invariant-cvl"
REPORT_TASK_ID = "report"


def bug_analysis_task_id(component_idx: int, slug: str) -> str:
    """task_id for a component's property-extraction ("bug analysis") phase."""
    return f"bug-{component_idx}-{slug}"


def cvl_gen_task_id(component_idx: int, slug: str) -> str:
    """task_id for a component's CVL-generation phase."""
    return f"cvl-{component_idx}-{slug}"
