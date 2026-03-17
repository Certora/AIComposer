"""
Extraction stage — single-shot LLM call to extract property groups and rules.

Given a triaged report, extracts the structured property groups, their
individual verification rules, and associates cloud run URLs using
resolved link metadata.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from composer.corpus.cloud_run import ResolvedLink
from composer.corpus.models import (
    ReportMetadata, ReportExtraction, PropertyGroupExtraction,
)
from composer.corpus.preprocess import PreprocessedPDF
from composer.corpus.triage import _pdf_document_block
from composer.templates.loader import load_jinja_template


async def extract_report(
    pdf: PreprocessedPDF,
    metadata: ReportMetadata,
    resolved_links: list[ResolvedLink],
    llm: BaseChatModel,
) -> ReportExtraction:
    """Extract property groups and rules from a security report.

    The LLM produces only the property groups; we assemble the full
    ReportExtraction by combining them with the already-known metadata.

    Args:
        pdf: Preprocessed PDF with base64 content.
        metadata: Already-extracted metadata from triage stage.
        resolved_links: Cloud run URLs resolved to rule names.
        llm: LLM instance (thinking should already be configured by caller).

    Returns:
        ReportExtraction with metadata and property groups.
    """
    bound = llm.with_structured_output(PropertyGroupExtraction)

    system_prompt = load_jinja_template(
        "corpus_extraction_prompt.j2",
        protocol_name=metadata.protocol_name,
        protocol_description=metadata.protocol_description,
        resolved_links=resolved_links,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=[_pdf_document_block(pdf)]),
    ]

    groups = await bound.ainvoke(messages)
    assert isinstance(groups, PropertyGroupExtraction)

    return ReportExtraction(
        metadata=metadata,
        properties=groups.properties,
    )
