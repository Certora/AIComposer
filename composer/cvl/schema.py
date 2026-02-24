from typing import Annotated, Literal, Protocol
from pydantic import BaseModel, Field, Discriminator

class ImportSpec(BaseModel):
    spec_file: str = Field("The (relative) path to the spec file to import")

class ContractImport(BaseModel):
    contract_name: str = Field("The name of the contract to import (e.g. ERCTokenA)")
    as_name: str = Field("The CVL identifier to bind to, e.g. 'tokenA'")

class MappingType(BaseModel):
    """
    A CVL ghost mapping type. NB this is different from a mapping in Solidity/Vyper
    """
    type: Literal["mapping"]
    key_type: "CVLType" = Field(description="The type of the keys. Only primitive types (uint256, etc.), `bytes`, and `string` are valid.")
    value_type: "CVLType" = Field(description="The codomain of the mapping.")

class ArrayType(BaseModel):
    """
    The type for dynamically sized arrays. The base type may *NOT* be mappings.
    """
    type: Literal["dyn_array"]
    base_type: "CVLType" = Field(description="The type of elements of the array")

class StaticArrayType(BaseModel):
    """
    The type for statically sized array of size `n`. The base type may *NOT* be mappings.
    """
    type: Literal["static_array"]
    base_type: "CVLType" = Field(description="The type of elements of the array")
    n: int = Field(description="The static number of elements in the array")

class ContractType(BaseModel):
    """
    A type declared in some contract. E.g. if contract Foo declares a struct `MyStruct`, this
    type represents `Foo.MyStruct`
    """
    type: Literal["contract_type"]
    host_contract: str = Field(description="The name of the contract (NOT the alias created via using)")
    type_name: str = Field(description="The type declared in `host_contract")

class StorageType(BaseModel):
    """
    The `storage` type, which represents the entire state of the blockchain.
    """
    type: Literal["storage_type"]

class PrimitiveType(BaseModel):
    """
    One of the built-in primitive types allowed by solidity. This includes all of the primitives
    of Solidity, plus `bytes` and `string`
    """
    type: Literal["primitive"]
    type_name: str = Field(description="The name of the type, e.g., `uint256` of `bool`")

CVLType = Annotated[
    MappingType | ArrayType | StaticArrayType | ContractType | StorageType | PrimitiveType,
    Discriminator("type")
]

DataLocation = Literal["calldata", "memory", "storage"]

class VMType(BaseModel):
    """
    A type from the EVM. These are simply CVL types extended with an optional
    data location, as found in Solidity. The Data location *must* be omitted
    if the type is a *Solidity* primitive (uint256, bool, etc.), and *must* be included if the type is
    a Solidity reference type (`bytes`, `string`, `uint[]`, etc.)
    """
    base_type: CVLType = Field(description="The type without data location")
    location: DataLocation | None = Field(description="The data location if the type is considered a reference type by Solidity, None otherwise")

class VMParam(BaseModel):
    """
    A parameter of a Solidity function described in the methods block
    """
    ty: VMType = Field(description="The type of the parameter")
    name: str | None = Field(description="The name of the parameter, or None if the parameter needn't be bound in the method body/proof block")

class TypeAndId(BaseModel):
    """
    An identifier + type declaration.
    """
    decl_type: CVLType = Field(description="The type of `id`")
    id: str = Field(description="The identifier to declare with `decl_type`")

# Literals
class BoolLiteral(BaseModel):
    type: Literal["bool_literal"]
    value: bool = Field(description="Boolean value")

class NumberLiteral(BaseModel):
    type: Literal["number_literal"]
    value: str = Field(description="Number as string in base 10 to preserve precision")

class StringLiteral(BaseModel):
    type: Literal["string_literal"]
    value: str = Field(description="String value")

class ArrayLiteral(BaseModel):
    type: Literal["array_literal"]
    elements: list["Expression"] = Field(description="Array elements")

# Identifiers (variables and special keywords)
class Identifier(BaseModel):
    type: Literal["identifier"]
    name: str = Field(description="Identifier name (variable, lastStorage, lastReverted, etc.)")
    annotation: Literal["old", "new"] | None = Field(description="Optional @old or @new annotation", default=None)

# Binary Operators
class BinaryOp(BaseModel):
    type: Literal["binary_op"]
    operator: Literal["add", "sub", "mul", "div", "mod", "exponent", "and", "or", "implies", "iff", 
                     "bw_or", "bw_xor", "bw_and", "bw_lshift", "bw_rshift", "bw_rshift_wzeros",
                     "eq", "ne", "lt", "le", "gt", "ge", "in"] = Field(description="Binary operator type")
    left: "Expression" = Field(description="Left operand")
    right: "Expression" = Field(description="Right operand")

# Unary Operators
class UnaryOp(BaseModel):
    type: Literal["unary_op"]
    operator: Literal["not", "bw_not", "uminus"] = Field(description="Unary operator type")
    operand: "Expression" = Field(description="Operand")

# Conditional Expression
class ConditionalExp(BaseModel):
    type: Literal["conditional"]
    condition: "Expression" = Field(description="Condition expression")
    then_expr: "Expression" = Field(description="Expression if condition is true")
    else_expr: "Expression" = Field(description="Expression if condition is false")

# Quantifiers
class QuantifierExp(BaseModel):
    type: Literal["quantifier"]
    is_forall: bool = Field(description="True for forall, false for exists")
    variable: TypeAndId = Field(description="Quantified variable")
    body: "Expression" = Field(description="Body expression")

# Sum Expressions
class SumExp(BaseModel):
    type: Literal["sum"]
    is_unsigned: bool = Field(description="True for unsigned sum (usum)")
    variables: list[TypeAndId] = Field(description="Sum variables")
    body: "Expression" = Field(description="Body expression")

# Array and Field Access
class ArrayAccess(BaseModel):
    type: Literal["array_access"]
    base: "Expression" = Field(description="Base expression")
    index: "Expression" = Field(description="Index expression")

class FieldAccess(BaseModel):
    type: Literal["field_access"]
    base: "Expression" = Field(description="Base expression")
    field_name: str = Field(description="Field name")

# Function Applications
class FunctionCall(BaseModel):
    type: Literal["function_call"]
    application: "FunctionApplication" = Field(description="A function application")

# Signature Literal
class SignatureLiteral(BaseModel):
    type: Literal["signature_literal"]
    method_ref: "MethodReference" = Field(description="Method reference")
    parameter_types: list[VMType] = Field(description="Parameter types")

Expression = Annotated[
    BoolLiteral | NumberLiteral | StringLiteral | ArrayLiteral | Identifier | BinaryOp | UnaryOp | 
    ConditionalExp | QuantifierExp | SumExp | ArrayAccess | FieldAccess | FunctionCall |  SignatureLiteral,
    Discriminator("type")
]

class FilteredBlock(BaseModel):
    """
    The filter expression `e` to apply to method parameter with name `method_param`
    """
    e: Expression = Field(description="The filtering expression. Must be a boolean expression, closed over `method_param`. " \
    "Only methods which pass this filter as used in the rule/invariant")
    method_param: str = Field(description="The name of a method parameter to a rule/invariant")

class FunctionApplication(BaseModel):
    """
    Any function like application. This is used for cast operators, built in functions, CVL functions, macros, and
    contract functions.
    """
    annotation: Literal["withrevert", "norevert"] | None = Field(description="Annotation indicating whether the function should revert. Must be None for any application which is not a CVL function or method application. norevert is the default, and can usually be omitted.")
    name: str = Field(description="The name of the function-like object to invoke.")
    host_contract: str | None = Field(description="The name of the contract declaring this method. None if this is not a contract application")
    params: list[Expression] = Field(description="The arguments to the function")
    state: str | None = Field(description="Optional state binding (AT clause)", default=None)


# Left-hand side expressions for assignments
class IdLhs(BaseModel):
    type: Literal["id"]
    name: str = Field(description="Variable identifier")

class ArrayAccessLhs(BaseModel):
    type: Literal["array_access"]
    base: "LeftHandSide" = Field(description="Base expression")
    index: Expression = Field(description="Array index expression")

class FieldAccessLhs(BaseModel):
    type: Literal["field_access"]
    base: "LeftHandSide" = Field(description="Base expression")
    field_name: str = Field(description="Field name")

LeftHandSide = Annotated[IdLhs | ArrayAccessLhs | FieldAccessLhs, Discriminator("type")]

# Commands
class DeclarationCmd(BaseModel):
    type: Literal["declaration"]
    variable: TypeAndId = Field(description="Variable declaration with type and name")
    initial_value: Expression | None = Field(description="Optional initial value for definition", default=None)

class AssignmentCmd(BaseModel):
    type: Literal["assignment"]
    left_hand_sides: list[LeftHandSide] = Field(description="Variables/locations being assigned to")
    expression: Expression = Field(description="Expression being assigned")

class HavocCmd(BaseModel):
    type: Literal["havoc"]
    targets: list[LeftHandSide] = Field(description="Expressions to havoc")
    assumption: Expression | None = Field(description="Optional assumption after havoc", default=None)

class AssumeCmd(BaseModel):
    type: Literal["assume"]
    expression: Expression = Field(description="Expression to assume")
    message: str = Field(description="Justification for the assumption")

class AssumeInvariantCmd(BaseModel):
    type: Literal["assume_invariant"]
    invariant_name: str = Field(description="Name of the invariant to assume")
    arguments: list[Expression] = Field(description="Arguments to the invariant")

class ApplyCmd(BaseModel):
    type: Literal["apply"]
    target: FunctionApplication = Field(description="Function/method application expression")

class ResetStorageCmd(BaseModel):
    type: Literal["reset_storage"]
    target: Expression = Field(description="Target expression for storage reset")

class AssertCmd(BaseModel):
    type: Literal["assert"]
    expression: Expression = Field(description="Expression to assert")
    message: str = Field(description="Message indicating what is being asserted")

class SatisfyCmd(BaseModel):
    type: Literal["satisfy"]
    expression: Expression = Field(description="Expression to satisfy")
    message: str = Field(description="Message indicating what should be satisfied")

class ReturnCmd(BaseModel):
    type: Literal["return"]
    values: list[Expression] = Field(description="Return values (empty for void return)")

class RevertCmd(BaseModel):
    type: Literal["revert"]
    message: str | None = Field(description="Optional revert message", default=None)

class Else(BaseModel):
    type: Literal["else"]
    commands: list["Command"] = Field(description="The commands to execute if the condition is false")

class IfCmd(BaseModel):
    type: Literal["if"]
    condition: Expression = Field(description="If condition")
    then_cmd: list["Command"] = Field(description="Commands to execute if condition is true")
    else_block: "ElseBlock | None" = Field(description="Optional else handling")

ElseBlock = Annotated[IfCmd | Else, Discriminator("type")]


class BlockCmd(BaseModel):
    type: Literal["block"]
    commands: list["Command"] = Field(description="List of commands in the block")

Command = Annotated[
    DeclarationCmd | AssignmentCmd | HavocCmd | AssumeCmd | AssumeInvariantCmd | ApplyCmd | 
    ResetStorageCmd | AssertCmd | SatisfyCmd | ReturnCmd | RevertCmd | IfCmd | BlockCmd,
    Discriminator("type")
]

class CodeBlock(BaseModel):
    commands: list[Command] = Field(description="List of commands that make up this code block")

class RuleBlock(BaseModel):
    """
    A single rule, with optional parameters
    """
    type: Literal["rule"]
    rule_name: str = Field(description="The name of the rule")
    rule_params: list[TypeAndId] = Field(description="The rule parameters (non-deterministic inputs)")
    filtered_block: list[FilteredBlock] | None = Field(description="Filters on any method parameters to the rule")
    block: CodeBlock = Field(description="The body of the rule")

class FunctionDef(BaseModel):
    """
    A native CVL function declaration with `name`
    """
    type: Literal["func_def"]
    name: str = Field(description="The name of the function as a valid cvl identifier")
    params: list[TypeAndId] = Field(description="The arguments to the function")
    return_value: list[CVLType] = Field(description="The type(s) of the return values. Empty for 'void' functions.")
    block: CodeBlock = Field(description="The body of the function.")

class MethodWithoutReturn(BaseModel):
    host_contract: str | None = Field("The host contract *type* of the method (*NOT* the using alias), or None, if the contract under verification should be used")
    name: str = Field(description="The name of the method")
    params: list["NamedVMParam"] = Field(description="The formal arguments to the method")

class MethodTarget(BaseModel):
    type: Literal["specific_method"]
    sig: MethodWithoutReturn = Field(description="A specific method with the given signature")

class GenericTarget(BaseModel):
    type: Literal["generic"]

class SpecialMethod(BaseModel):
    type: Literal["special"]
    special_target: Literal["constructor", "fallback"]

ProofTarget = Annotated[
    SpecialMethod | GenericTarget | MethodTarget, Discriminator("type")
]

class WithBlock(BaseModel):
    """
    Binds the environment in a preserved block/summary.
    """
    id: str = Field(description="The name to bind to the environment (of type `env`)")

class ProofBlock(BaseModel):
    """
    A preserved of "proof block" that controls how the invariant is checked
    for given methods
    """
    target: ProofTarget = Field(description="The target of the proof block")
    block: CodeBlock = Field(description="The block to control the checking of the invariant. Must be closed under the method signature args (if any), the invariant params, and the `with` block (if any).")
    with_binding: WithBlock | None = Field(description="An optional binding for the environment used to invoke the method for this proof block.")

class Invariant(BaseModel):
    """
    Invariant declaration
    """
    type: Literal["invariant"]
    name: str = Field("The name of the invariant")
    invariant_params: list[TypeAndId] = Field(description="The (non-deterministically chosen) arguments to the invariant.")
    invariant_expression: Expression = Field(description="The expression to show is invariant. Should be boolean typed.")
    filter: FilteredBlock | None = Field(description="The filter for the invariant. The `method_param` name can be arbitrarily chosen, it is always bound" \
    "to the method being considered for invariant checking")
    proofs: list[ProofBlock] = Field(description="Optional proof blocks for methods that require special pre-condition handling")

class SortDef(BaseModel):
    """
    Declare an uninterpreted sort.
    """
    type: Literal["sort_decl"]
    sort_name: str = Field(description="The name of the uninterpreted sort")

class GhostFunction(BaseModel):
    """
    The type of ghost functions. This is considered slightly deprecated; in most
    cases a ghost mapping is appropriate.
    """
    type: Literal["ghost_fun"]
    params: list[CVLType] = Field(description="The arguments to the ghost function.")
    result_type: CVLType = Field(description="The (single) type returned by the function")

class BasicGhostType(BaseModel):
    type: Literal["ghost_type"]
    base_type: Annotated[PrimitiveType | MappingType, Discriminator("type")] = Field(description="A basic type. Only primitives and mappings can be used.")

GhostType = Annotated[GhostFunction | BasicGhostType, Discriminator("type")]

class GhostAxiom(BaseModel):
    initial: bool = Field(description="Whether this axiom only holds on the initial state (true), or at all points (false)")
    exp: Expression = Field(description="A boolean expression. Must be closed under the ghost name being axiomatized.")

class GhostDef(BaseModel):
    type: Literal["ghost_def"]
    ghost_name: str = Field(description="The name of the ghost")
    ghost_type: GhostType = Field(description="The type of the ghost")
    persistent: bool = Field(description="Whether the ghost is persistent, i.e., if true, reverts do not rollback the state of the ghosts.")
    axioms: list[GhostAxiom] = Field(description="Axioms that must hold for the ghost.")

class MacroDef(BaseModel):
    """
    A macro definition
    """
    type: Literal["macro_def"]
    id: str = Field(description="The name of the macro being defined")
    params: list[TypeAndId] = Field(description="The parameters to the macro. An empty list defines a global constant")
    result_type: CVLType = Field(description="The result type of the macro")
    exp: Expression = Field(description="The body of the macro, as an expression yielding a value of type `result_type`")

# Hook Parameter Types  
class NamedVMParam(BaseModel):
    param_type: CVLType = Field(description="The (Solidity) type of the parameter, *without* a data location")
    name: str = Field(description="The parameter name")

# Slot Pattern Types (excluding deprecated slot/offset syntax)
class NamedSlotPattern(BaseModel):
    type: Literal["named"]
    name: str = Field(description="Named identifier for the slot pattern")

class MapAccessSlotPattern(BaseModel):
    type: Literal["map_access"] 
    base_pattern: "SlotPattern" = Field(description="The base slot pattern")
    key_param: NamedVMParam = Field(description="Parameter for the mapping key")

class ArrayAccessSlotPattern(BaseModel):
    type: Literal["array_access"]
    base_pattern: "SlotPattern" = Field(description="The base slot pattern") 
    index_param: NamedVMParam = Field(description="Parameter for the array index")

class FieldAccessSlotPattern(BaseModel):
    type: Literal["field_access"]
    base_pattern: "SlotPattern" = Field(description="The base slot pattern")
    field_name: str = Field(description="Name of the field being accessed")

SlotPattern = Annotated[
    NamedSlotPattern | MapAccessSlotPattern | ArrayAccessSlotPattern | FieldAccessSlotPattern,
    Discriminator("type")
]

class StatePattern(Protocol):
    slot_pattern: SlotPattern
    type: str

class LoadPattern(StatePattern):
    value_param: NamedVMParam

class StorePattern(StatePattern):
    new_value_param: NamedVMParam

    old_value_param: NamedVMParam | None


# Hook Definitions
class SloadHook(BaseModel):
    type: Literal["sload"]
    value_param: NamedVMParam = Field(description="Parameter that will hold the loaded value")
    slot_pattern: SlotPattern = Field(description="Pattern matching the storage slot")

class SstoreHook(BaseModel):
    type: Literal["sstore"]
    slot_pattern: SlotPattern = Field(description="Pattern matching the storage slot")
    new_value_param: NamedVMParam = Field(description="Parameter that will hold the new value being stored")
    old_value_param: NamedVMParam | None = Field(description="Optional parameter that will hold the previous value", default=None)

class TloadHook(BaseModel):
    type: Literal["tload"]
    value_param: NamedVMParam = Field(description="Parameter that will hold the loaded transient value")
    slot_pattern: SlotPattern = Field(description="Pattern matching the transient storage slot")

class TstoreHook(BaseModel):
    type: Literal["tstore"] 
    slot_pattern: SlotPattern = Field(description="Pattern matching the transient storage slot")
    new_value_param: NamedVMParam = Field(description="Parameter that will hold the new value being stored")
    old_value_param: NamedVMParam | None = Field(description="Optional parameter that will hold the previous value", default=None)

class CreateHook(BaseModel):
    type: Literal["create"]
    param: NamedVMParam = Field(description="The address of the created contract generated by the CREATE opcode being hooked upon")

class OpcodeHook(BaseModel):
    type: Literal["opcode"]
    opcode_name: str = Field(description="Name of the opcode to hook")
    input_params: list[NamedVMParam] = Field(description="Input parameters for the opcode")
    output_param: NamedVMParam | None = Field(description="Optional output parameter", default=None)

HookPattern = Annotated[
    SloadHook | SstoreHook | TloadHook | TstoreHook | CreateHook | OpcodeHook,
    Discriminator("type")
]

class HookDef(BaseModel):
    type: Literal["hook_def"]
    pattern: HookPattern = Field(description="The hook pattern defining what to intercept")
    block: CodeBlock = Field(description="The CVL code block to execute when the hook is triggered. Can access the identifiers bound in the pattern")

class UseDirective(BaseModel):
    pass

class OverrideDirective(BaseModel):
    pass

# Method References
class MethodReference(BaseModel):
    contract: str | None = Field(description="Contract name, None for current contract, or `_` for all methods with a matching signature in any contract")
    method_name: str = Field(description="Method name")

# Expect Clauses (when present)
class ExpectType(BaseModel):
    type: Literal["type"]
    expected_types: list[VMType] = Field(description="Expected return types")

class ExpectVoid(BaseModel):
    type: Literal["void"]

ExpectClause = Annotated[ExpectType | ExpectVoid, Discriminator("type")]

class HavocingSummary(BaseModel):
    type: Literal["havocing"]
    havoc_keyword: Literal["nondet", "havoc_all", "havoc_ecf", "auto", "assert_false"]

# Call Summaries
class KeywordSummary(BaseModel):
    type: Literal["keyword"]
    summary_keyword: Literal["constant", "per_callee_constant"] = Field(description="The summary keyword")

class AlwaysSummary(BaseModel):
    type: Literal["always"]
    expression: Expression = Field(description="(Constant) expression to use as the result of the method")

class ExpressionSummary(BaseModel):
    type: Literal["expression"]
    expression: Expression = Field(description="Summary expression")
    expect_clause: ExpectClause | None = Field(description="Optional expected return behavior", default=None)

class DispatcherSummary(BaseModel):
    type: Literal["dispatcher"]
    optimistic: bool = Field(description="Whether the dispatcher is optimistic, i.e., assume one of the dispatchees is inlined")
    use_fallback: bool = Field(description="Whether to use fallback functions as a candidate to which to dispatch")

CallSummary = Annotated[
    KeywordSummary | AlwaysSummary | ExpressionSummary | DispatcherSummary | HavocingSummary,
    Discriminator("type")
]

PreReturnFlag = Literal["internal", "external"]

PostReturnFlags = Literal["optional", "envfree"]

# Method Signature
class MethodSignature(BaseModel):
    method_ref: MethodReference = Field(description="Reference to the method")
    parameters: list[VMParam] = Field(description="Method parameters")
    return_types: list[VMParam] = Field(description="Return types (empty for void methods or for `_.methodName` entries)")
    visibility: PreReturnFlag = Field(description="Identifies this method as external vs internal")
    post_flags: list[PostReturnFlags] = Field(description="Post-return flags")

# Method Entries
class ImportedFunction(BaseModel):
    type: Literal["imported_function"]
    signature: MethodSignature = Field(description="Method signature")
    summary: CallSummary | None = Field(description="Optional call summary", default=None)
    with_env: str | None = Field(description="The name of the bound environment variable, if any", default=None)

class CatchAllSummary(BaseModel):
    type: Literal["catch_all"]
    contract_name: str = Field(description="The name of the contract whose external methods should have this summary")
    summary: HavocingSummary = Field(description="Summary to apply to all matching methods")

class AllMethodsInContract(BaseModel):
    """
    Include all methods in the target contract
    """
    type: Literal["all_in_contract"]
    contract_name: str = Field(description="The contract whose methods to include")

class SpecificMethodSignature(BaseModel):
    """
    A method signature `sig = method_name(params)`.
    If `host_contract` is None, refers to any method with matching the signature, otherwise
    refers to `host_contract.method_name(params)`
    """
    type: Literal["matching_signature"]
    host_contract: str | None = Field(description="If non-none, include only the method with signature `sig` in the given contract. Must be the name of the contract type, *NOT* the imported alias." \
      " None means all methods with that signature.")
    method_name: str = Field(description="The name of the method")
    params: list[VMType] = Field(description="The parameter types of the method")

DispatchPattern = Annotated[
    SpecificMethodSignature | AllMethodsInContract, Discriminator("type")
]

class DispatchList(BaseModel):
    optimistic: bool = Field(description="Whether the summari should *assume* that the call resolves to one of the functions included in this list")
    use_fallback: bool = Field(description="Whether the fallback function of a target contract (if it exists) should be included in the dispatch chain")
    patterns: list[DispatchPattern] = Field(description="The methods to include in the dispatch chain")
    default_summary: HavocingSummary | None = Field(description="The summary to use in the fallthrough case if the summary is not optimistic.")

class AnyMatchingSignature(BaseModel):
    """
    Represents _.method(params)
    """
    type: Literal["with_signature"]
    method_name: str = Field(description="The name of the method")
    params: list[VMParam] = Field(description="The parameters of the method")

class SpecificMethod(BaseModel):
    """
    Represents Contract.method(params)
    """
    type: Literal["specific_method"]
    host_contract: str | None = Field(description="The host type of the method (not the alias); None if the contract under verification should be used")
    param_types: list[VMParam] = Field(description="The parameters to the function")
    name: str = Field(description="The name of the method")

class UnresolvedEverywhere(BaseModel):
    """
    Represents all methods in all contract, i.e. _._ in the surface syntax.
    """
    type: Literal["catch_all"]

UnresolvedInTarget = Annotated[
    UnresolvedEverywhere | SpecificMethod | AnyMatchingSignature | AllMethodsInContract, Discriminator("type")
]

class UnresolvedDynamicSummary(BaseModel):
    type: Literal["unresolved_dynamic"]
    method_ref: UnresolvedInTarget = Field(description="Where this unresolved summary should apply")
    summary: DispatchList = Field(description="The dispatch list to use")

MethodEntry = Annotated[
    ImportedFunction | CatchAllSummary | UnresolvedDynamicSummary,
    Discriminator("type")
]

class MethodsBlock(BaseModel):
    type: Literal["methods_block"]
    method_entries: list[MethodEntry] = Field(description="List of method declarations and summaries")

BasicBlock = Annotated[
    Invariant | FunctionDef | RuleBlock | SortDef | GhostDef | MacroDef | HookDef | MethodsBlock,
    Discriminator("type")
]

class CVLFile(BaseModel):
    #import_specs: list[ImportSpec] = Field(description="A list of spec files to import")
    import_contract: list[ContractImport] = Field(description="A list of contracts to bind in the CVL namespace")
    blocks: list[BasicBlock] = Field(description="The top-level blocks making up the spec file")