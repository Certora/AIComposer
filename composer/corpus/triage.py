"""
Triage stage — single-shot LLM call to classify a security report.

Determines report type (formal verification vs manual audit), extracts
protocol name/description, and finds repository references from the PDF.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from composer.corpus.models import ReportMetadata
from composer.corpus.preprocess import PreprocessedPDF


_TRIAGE_SYSTEM_PROMPT = """\
You are classifying a security report from Certora.

Your job is to extract high-level metadata from the report:
1. **Protocol name** — the name of the protocol or project being audited.
2. **Protocol description** — a brief (1-2 sentence) description of what the protocol does.
3. **Report type**:
   - "formal_verification" if the report describes Certora Prover / CVL-based formal verification
     for a protocol deployed on the EVM
     (look for terms like "invariant", "rule", "CVL", "Certora Prover", "formally verified").
   - "not_evm" if the report describes formal verification for a different blockchain (Solana/Stellar/etc.)
   - "manual_audit" if it's a traditional code review / manual security audit.
   - "other" for anything else.
4. **Repository reference** — if the report mentions a GitHub repository URL and/or commit hash,
   extract them.
"""


def _pdf_document_block(pdf: PreprocessedPDF) -> dict:
    """Build a langchain document content block from a preprocessed PDF."""
    return {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": pdf.pdf_base64,
        },
    }


async def triage_report(pdf: PreprocessedPDF, llm: BaseChatModel) -> ReportMetadata:
    """Classify a security report and extract metadata.

    Args:
        pdf: Preprocessed PDF with base64 content.
        llm: LLM instance (thinking should already be configured by caller).

    Returns:
        ReportMetadata with classification and extracted info.
    """
    bound = llm.with_structured_output(ReportMetadata)

    messages = [
        SystemMessage(content=_TRIAGE_SYSTEM_PROMPT),
        HumanMessage(content=[_pdf_document_block(pdf)]),
    ]

    result = await bound.ainvoke(messages)
    assert isinstance(result, ReportMetadata)
    return result
