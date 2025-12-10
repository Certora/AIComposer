# AI Composer Flow Diagram

This document provides a visual overview of the AI Composer workflow to help code contributors understand the system architecture and execution flow. The AI Composer generates verified smart contract implementations from documentation and CVL (Certora Verification Language) specifications.

## High-Level Flow

```mermaid
flowchart TD
    Start([main.py: Entry Point]) --> ParseArgs[Parse Command Line Arguments]
    ParseArgs --> SetupLogging[Setup Logging & Debug Handler]
    SetupLogging --> CreateLLM[Create LLM Instance<br/>ChatAnthropic with Claude]
    CreateLLM --> CheckDebugFS{Debug FS<br/>Mode?}
    
    CheckDebugFS -->|Yes| DumpFS[Dump Virtual Filesystem<br/>and Exit]
    CheckDebugFS -->|No| UploadInput[Upload Input Files<br/>spec_file, interface_file, system_doc]
    
    UploadInput --> ExecuteWorkflow[Execute AI Composer Workflow]
    
    ExecuteWorkflow --> InitSetup[Initialize Setup]
    InitSetup --> CheckInputType{Input Type?}
    
    CheckInputType -->|InputData| FreshInput[Fresh Workflow:<br/>Load spec, interface, system_doc]
    CheckInputType -->|ResumeIdData| ResumeId[Resume from ID:<br/>Load from audit DB]
    CheckInputType -->|ResumeFSData| ResumeFS[Resume from FS:<br/>Load from directory]
    
    FreshInput --> ExtractReqs{Requirements<br/>Enabled?}
    ResumeId --> ExtractReqs
    ResumeFS --> ExtractReqs
    
    ExtractReqs -->|Yes| ReqExtraction[Extract Natural Language<br/>Requirements from System Doc]
    ExtractReqs -->|No| BuildWorkflow[Build Workflow Graph]
    ReqExtraction --> BuildWorkflow
    
    BuildWorkflow --> RegisterRun[Register Run in Audit DB]
    RegisterRun --> CompileGraph[Compile Workflow Graph]
    CompileGraph --> StreamLoop[Stream Workflow Execution]
    
    StreamLoop --> WorkflowGraph[Workflow Graph Execution]
    
    WorkflowGraph --> CheckCompletion{Generated<br/>Code?}
    CheckCompletion -->|No| ContinueLoop[Continue Loop]
    CheckCompletion -->|Yes| Validate[Validate Requirements<br/>& Prover Results]
    
    Validate --> PrintResults[Print Generated Code]
    PrintResults --> End([End])
    
    ContinueLoop --> WorkflowGraph
    
    style Start fill:#e1f5ff
    style End fill:#e1f5ff
    style WorkflowGraph fill:#fff4e1
    style Validate fill:#e8f5e9
```

## Workflow Graph Structure

The core workflow graph follows a standard LangGraph pattern with the following nodes:

```mermaid
flowchart LR
    INITIAL[initial Node<br/>Send System Prompt<br/>& Initial Instructions] --> TOOLS[tools Node<br/>Execute LLM Tool Calls]
    
    TOOLS --> CheckEnd{Output<br/>Key Set?}
    CheckEnd -->|Yes| END([__end__])
    CheckEnd -->|No| CheckSummarize{Messages ><br/>Threshold?}
    
    CheckSummarize -->|Yes| SUMMARIZE[summarize Node<br/>Summarize History]
    CheckSummarize -->|No| TOOL_RESULT[tool_result Node<br/>Process Tool Results<br/>& Continue LLM]
    
    SUMMARIZE --> TOOL_RESULT
    TOOL_RESULT --> TOOLS
    
    style INITIAL fill:#e3f2fd
    style TOOLS fill:#fff3e0
    style TOOL_RESULT fill:#f3e5f5
    style SUMMARIZE fill:#e8f5e9
    style END fill:#ffebee
```

**How it works:**

LangGraph is a framework for building stateful, multi-actor applications with LLMs. The workflow follows a simple loop pattern:

1. **initial Node**: Runs once at the start, sends the system prompt and initial instructions to the LLM.
2. **tools Node**: Executes any tool calls that the LLM requested (e.g., `certora_prover`, `PutFile`, etc.).
3. **Routing Decision**: After tools execute, the graph checks:
   - If `generated_code` is set → workflow ends
   - If message count exceeds threshold → route to `summarize` (optional, reduces context size)
   - Otherwise → route to `tool_result`
4. **tool_result Node**: Sends tool results back to the LLM, which then decides what to do next (call more tools, fix code, or complete).
5. **Loop**: The graph loops back to `tools` node, creating an iterative cycle until the LLM calls the `code_result` tool to signal completion.

This pattern allows the AI Composer to iteratively write code, verify it with the prover, fix issues, and repeat until all specifications are satisfied.

## Available Tools

The AI Composer has access to the following tools during execution:

```mermaid
graph TB
    Tools[Available Tools] --> CoreTools[Core Tools]
    Tools --> VFSTools[VFS Tools]
    Tools --> ConditionalTools[Conditional Tools]
    
    CoreTools --> Prover[certora_prover<br/>Run Certora Prover<br/>to verify contracts]
    CoreTools --> SpecChange[propose_spec_change<br/>Propose spec modifications<br/>requires human approval]
    CoreTools --> HumanLoop[human_in_the_loop<br/>Request human assistance]
    CoreTools --> CodeResult[code_result<br/>Mark completion<br/>submit final code]
    CoreTools --> CVLSearch[cvl_manual_search<br/>Search CVL documentation]
    
    VFSTools --> PutFile[PutFile<br/>Write files to VFS]
    VFSTools --> GetFile[GetFile<br/>Read files from VFS]
    VFSTools --> ListDir[ListDirectory<br/>List directory contents]
    VFSTools --> DeleteFile[DeleteFile<br/>Remove files from VFS]
    
    ConditionalTools --> JudgeTool[judge_tool<br/>Validate requirements<br/>if requirements enabled]
    ConditionalTools --> Relaxation[requirements_relaxation<br/>Request requirement relaxation<br/>if requirements enabled]
    ConditionalTools --> Memory[memory<br/>Persistent memory tool<br/>if beta features enabled]
    
    style CoreTools fill:#e3f2fd
    style VFSTools fill:#fff3e0
    style ConditionalTools fill:#f3e5f5
```

## Tool Execution Flow

```mermaid
sequenceDiagram
    participant AIComposer as AI Composer (LLM)
    participant Tools as Tools Node
    participant Prover as certora_prover
    participant VFS as Virtual File System
    participant Result as tool_result Node
    
    AIComposer->>Tools: Tool Call Request
    Tools->>Prover: Execute certora_prover
    Prover->>VFS: Read source files
    Prover->>Prover: Run Certora Prover
    Prover->>Tools: Return Report/Results
    Tools->>Result: Tool Message
    Result->>AIComposer: Process Results & Continue
    AIComposer->>AIComposer: Analyze Results & Decide Next Action
    
    alt All Rules Verified
        AIComposer->>Tools: code_result tool
        Tools->>Result: Set generated_code
        Result->>AIComposer: Workflow Complete
    else Rules Violated
        AIComposer->>VFS: PutFile (fix code)
        AIComposer->>Tools: certora_prover (re-verify)
    else Spec Needs Change
        AIComposer->>Tools: propose_spec_change
        Tools->>Human: Interrupt for Approval
        Human->>Tools: ACCEPTED/REJECTED/REFINE
    end
```

## Requirements Extraction Flow (Optional)

When requirements are enabled, the system first extracts natural language requirements:

```mermaid
flowchart TD
    StartReq[Start Requirements<br/>Extraction] --> LoadDocs[Load System Doc<br/>& Spec File]
    LoadDocs --> BuildReqGraph[Build Requirements<br/>Extraction Workflow]
    BuildReqGraph --> ReqStream[Stream Requirements<br/>Extraction]
    
    ReqStream --> ReqLLM[LLM Analyzes Documents]
    ReqLLM --> ReqTools{Use Tools?}
    
    ReqTools -->|cvl_manual_search| ReqSearch[Search CVL Manual]
    ReqTools -->|human_in_the_loop| ReqHuman[Ask Human Questions]
    ReqTools -->|memory| ReqMemory[Store/Retrieve Memories]
    
    ReqSearch --> ReqLLM
    ReqHuman --> ReqLLM
    ReqMemory --> ReqLLM
    
    ReqTools -->|results_tool| ReqComplete[Extract Requirements List]
    ReqComplete --> StoreReqs[Store Requirements<br/>in Store]
    StoreReqs --> AddJudgeTool[Add judge_tool<br/>& relaxation_tool]
    AddJudgeTool --> EndReq[End Requirements<br/>Extraction]
    
    style StartReq fill:#e1f5ff
    style EndReq fill:#e1f5ff
    style ReqLLM fill:#fff4e1
```

## Validation & Completion Flow

Before the workflow completes, the system validates that all requirements are met:

```mermaid
flowchart TD
    CodeResult[code_result Tool Called] --> CheckValidation{Check Validation<br/>State}
    
    CheckValidation --> CheckProver{Prover<br/>Validated?}
    CheckProver -->|No| RejectProver[REJECT: Prover<br/>not validated]
    CheckProver -->|Yes| CheckReqs{Requirements<br/>Validated?}
    
    CheckReqs -->|No| RejectReqs[REJECT: Requirements<br/>not validated]
    CheckReqs -->|Yes| Accept[ACCEPT: All<br/>Validations Pass]
    
    RejectProver --> ContinueWork[Continue Workflow<br/>with Error Message]
    RejectReqs --> ContinueWork
    ContinueWork --> WorkflowGraph[Return to Workflow]
    
    Accept --> ExtractCode[Extract Generated Code<br/>from VFS]
    ExtractCode --> PrintOutput[Print Source Files<br/>& Comments]
    PrintOutput --> Complete([Workflow Complete])
    
    style CodeResult fill:#e3f2fd
    style Accept fill:#e8f5e9
    style RejectProver fill:#ffebee
    style RejectReqs fill:#ffebee
    style Complete fill:#e1f5ff
```

## Key Components

### State Management
- **AIComposerState**: Main state containing messages, VFS, validation status, and generated_code
- **AIComposerContext**: Context with LLM, RAG DB, prover options, VFS materializer, and required validations
- **Input**: Input schema with input messages and initial VFS state

### Persistence
- **PostgresSaver**: Checkpoints workflow state for resumption
- **PostgresStore**: Stores requirements and other metadata
- **AuditDB**: Tracks runs, artifacts, and completion status
- **PostgresMemoryBackend**: Persistent memory for requirements and composer context

### Key Files
- `main.py`: Entry point
- `composer/workflow/executor.py`: Main workflow execution logic
- `composer/workflow/factories.py`: Workflow graph builder
- `graphcore/graph.py`: Core workflow graph construction
- `composer/tools/`: Individual tool implementations
- `composer/natreq/extractor.py`: Requirements extraction
- `composer/natreq/judge.py`: Requirements validation

## Resume Workflow

The system supports resuming workflows from checkpoints:

```mermaid
flowchart LR
    Resume[Resume Command] --> CheckType{Resume Type?}
    
    CheckType -->|resume-id| LoadFromDB[Load Artifact<br/>from Audit DB]
    CheckType -->|resume-dir| LoadFromFS[Load from<br/>File System]
    
    LoadFromDB --> GetCheckpoint[Get Checkpoint ID]
    LoadFromFS --> GetCheckpoint
    
    GetCheckpoint --> RestoreState[Restore Workflow State]
    RestoreState --> Continue[Continue Execution]
    
    style Resume fill:#e1f5ff
    style Continue fill:#e8f5e9
```

## Human Interaction Points

The workflow can be interrupted for human input at several points:

1. **propose_spec_change**: When the AI Composer wants to modify the specification
2. **human_in_the_loop**: When the AI Composer needs clarification or assistance
3. **Requirements Extraction**: When extracting requirements, the AI Composer may ask questions
4. **Debug Handler**: Ctrl+C during execution to access debug console
