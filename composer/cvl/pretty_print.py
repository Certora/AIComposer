"""
Pretty printer for CVL schema with proper precedence handling.

Precedence levels (from lowest to highest, based on the CVL grammar):
0. Conditional (?:)
1. IFF (<=>)
2. Implies (=>)
3. Logical OR (||)
4. Logical AND (&&)
5. Relational (==, !=, <, <=, >, >=)
6. Bitwise OR (|)
7. Bitwise XOR (^)
8. Bitwise AND (&)
9. Bitwise shifts (<<, >>, >>>)
10. Additive (+, -)
11. Multiplicative (*, /, %)
12. Unary (!, ~, -)
13. Exponentiation (**)
14. Set membership (in)
15. Postfix (array access, field access, function calls)
"""

from typing import Callable, TypeVar, cast
from composer.cvl.schema import *

T = TypeVar("T")

# Precedence levels for expression types
PRECEDENCE = {
    # Lowest precedence
    "conditional": 0,
    
    # Binary operators by precedence level
    "iff": 1,
    "implies": 2, 
    "or": 3,
    "and": 4,
    "eq": 5, "ne": 5, "lt": 5, "le": 5, "gt": 5, "ge": 5,
    "bw_or": 6,
    "bw_xor": 7,
    "bw_and": 8,
    "bw_lshift": 9, "bw_rshift": 9, "bw_rshift_wzeros": 9,
    "add": 10, "sub": 10,
    "mul": 11, "div": 11, "mod": 11,
    "exponent": 13,
    "in": 14,
    
    # Unary operators
    "not": 12, "bw_not": 12, "uminus": 12,
    
    # Postfix and primary expressions (highest precedence)
    "array_access": 15,
    "field_access": 15,
    "function_call": 15,
    
    # Literals and identifiers (highest precedence)
    "bool_literal": 16,
    "number_literal": 16,
    "string_literal": 16,
    "array_literal": 16,
    "identifier": 16,
    "signature_literal": 16,
    
    # Special cases
    "quantifier": 0,  # Low precedence like conditional
    "sum": 0,         # Low precedence like conditional
}

def get_precedence(expr: Expression) -> int:
    """Get the precedence level of an expression."""
    if expr.type == "binary_op":
        return PRECEDENCE.get(expr.operator, 0)
    elif expr.type == "unary_op":
        return PRECEDENCE.get(expr.operator, 12)
    elif expr.type in PRECEDENCE:
        return PRECEDENCE[expr.type]
    else:
        raise ValueError(f"Unknown expression type {expr.type}")

def needs_parentheses(expr: Expression, parent_expr: Expression, is_right_operand: bool = False) -> bool:
    """
    Determine if an expression needs parentheses when printed within a parent expression.
    
    Args:
        expr: The expression being printed
        parent_expr: The parent expression containing expr
        is_right_operand: True if expr is the right operand of a binary operator
    """
    expr_prec = get_precedence(expr)
    parent_prec = get_precedence(parent_expr)
    
    # Lower precedence expressions need parentheses
    if expr_prec < parent_prec:
        return True
    
    # Same precedence: check associativity
    if expr_prec == parent_prec:
        if parent_expr.type == "binary_op":
            # Right-associative operators
            if parent_expr.operator in ["implies", "iff", "exponent"]:
                return not is_right_operand
            # Left-associative operators
            else:
                return is_right_operand
    
    return False

class LineInterleaver:
    def __init__(self, parent: "LineBuilder"):
        self.parent = parent
        self.string_buff = ""
    def __enter__(self) -> "LineInterleaver":
        return self
    def __exit__(self, *args):
        self.parent.finalize_me(self)
    def append(self, content: str):
        self.string_buff += content

class LineBuilder:
    def __init__(self, indent_amt: int = 0, parent: "LineBuilder | None" = None):
        self.indent_amt = indent_amt
        self.parent = parent
        self.buffer : list[str] = []
        self.child_line : LineInterleaver | None = None
    def indent(self) -> "LineBuilder":
        if self.child_line is not None and self.child_line.string_buff:
            self.line(self.child_line.string_buff)
            self.child_line.string_buff = ""
        return LineBuilder(self.indent_amt + 1, parent = self)
    def __enter__(self) -> "LineBuilder":
        return self
    def __exit__(self, *args):
        if self.parent is not None:
            self.parent.buffer.extend(self.buffer)
    def finalize_me(self, interleaver: LineInterleaver):
        if interleaver.string_buff:
            self.line(interleaver.string_buff)
        self.child_line = None
    def line(self, line: str):
        self.buffer.append((self.indent_amt * "  ") + line)
    def line_builder(self) -> LineInterleaver:
        if self.child_line is not None:
            raise RuntimeError("Cannot build multiple lines at once")
        to_ret = LineInterleaver(self)
        self.child_line = to_ret
        return to_ret
        

class CVLPrettyPrinter:
    """Pretty printer for CVL expressions and schema elements."""

    def print_expression(self, expr: Expression, parent_expr: Expression | None = None, is_right_operand: bool = False) -> str:
        """Print an expression, adding parentheses if needed."""
        if parent_expr and needs_parentheses(expr, parent_expr, is_right_operand):
            return f"({self._print_expression_inner(expr)})"
        return self._print_expression_inner(expr)
    
    def _print_expression_inner(self, expr: Expression) -> str:
        """Print an expression without considering parentheses."""
        match expr.type:
            case "bool_literal":
                return self._print_bool_literal(expr)
            case "number_literal":
                return self._print_number_literal(expr)
            case "string_literal":
                return self._print_string_literal(expr)
            case "array_literal":
                return self._print_array_literal(expr)
            case "identifier":
                return self._print_identifier(expr)
            case "binary_op":
                return self._print_binary_op(expr)
            case "unary_op":
                return self._print_unary_op(expr)
            case "conditional":
                return self._print_conditional(expr)
            case "quantifier":
                return self._print_quantifier(expr)
            case "sum":
                return self._print_sum(expr)
            case "array_access":
                return self._print_array_access(expr)
            case "field_access":
                return self._print_field_access(expr)
            case "function_call":
                return self._print_function_call(expr)
            case "signature_literal":
                return self._print_signature_literal(expr)

    def _print_bool_literal(self, expr: BoolLiteral) -> str:
        return "true" if expr.value else "false"
    
    def _print_number_literal(self, expr: NumberLiteral) -> str:
        return expr.value
    
    def _print_string_literal(self, expr: StringLiteral) -> str:
        return f'"{expr.value}"'
    
    def _print_array_literal(self, expr: ArrayLiteral) -> str:
        elements = [self.print_expression(elem) for elem in expr.elements]
        return f'[{", ".join(elements)}]'
    
    def _print_identifier(self, expr: Identifier) -> str:
        to_ret = expr.name
        if expr.annotation is not None:
            to_ret += f"@{expr.annotation}"
        return to_ret
        
    def _print_binary_operator(self, expr: BinaryOp) -> str:
        match expr.operator:
            case "add":
                return "+"
            case "and":
                return "&&"
            case "bw_and":
                return "&"
            case "bw_or":
                return "|"
            case "bw_lshift":
                return "<<"
            case "bw_rshift":
                return ">>"
            case "bw_rshift_wzeros":
                return ">>>"
            case "bw_xor":
                return "xor"
            case "div":
                return "/"
            case "exponent":
                return "^"
            case "eq":
                return "=="
            case "ge":
                return ">="
            case "gt":
                return ">"
            case "iff":
                return "<=>"
            case "implies":
                return "=>"
            case "in":
                return "in"
            case "le":
                return "<="
            case "lt":
                return "<"
            case "mod":
                return "%"
            case "mul":
                return "*"
            case "ne":
                return "!="
            case "or":
                return "||"
            case "sub":
                return "-"            
        
    
    def _print_binary_op(self, expr: BinaryOp) -> str:
        left = self.print_expression(expr.left, expr, False)
        right = self.print_expression(expr.right, expr, True)        
        operator = self._print_binary_operator(expr)
        return f"{left} {operator} {right}"
    
    def _print_unary_operator(self, expr: UnaryOp) -> str:
        match expr.operator:
            case "bw_not":
                return "~"
            case "not":
                return "!"
            case "uminus":
                return "-"
    
    def _print_unary_op(self, expr: UnaryOp) -> str:
        operand = self.print_expression(expr.operand, expr)
        operator = self._print_unary_operator(expr)
        return f"{operator} {operand}"
    
    def _print_conditional(self, expr: ConditionalExp) -> str:
        condition = self.print_expression(expr.condition, expr)
        then_expr = self.print_expression(expr.then_expr, expr)
        else_expr = self.print_expression(expr.else_expr, expr)
        return f"{condition} ? {then_expr} : {else_expr}"
    
    def _print_quantifier(self, expr: QuantifierExp) -> str:
        var = self._print_type_and_id(expr.variable)
        body = self.print_expression(expr.body, expr)
        quantifier = "forall" if expr.is_forall else "exists"
        return f"{quantifier} {var}.{body}"
    
    def _print_sum(self, expr: SumExp) -> str:
        vars = [self._print_type_and_id(var) for var in expr.variables]
        body = self.print_expression(expr.body, expr)
        keyword = "usum" if expr.is_unsigned else "sum"
        return f"{keyword} {', '.join(vars)}.{body}"
    
    def _print_array_access(self, expr: ArrayAccess) -> str:
        base = self.print_expression(expr.base, expr)
        index = self.print_expression(expr.index)
        return f"{base}[{index}]"
    
    def _print_field_access(self, expr: FieldAccess) -> str:
        base = self.print_expression(expr.base, expr)
        return f"{base}.{expr.field_name}"
    
    def _print_function_call(self, expr: FunctionCall) -> str:
        app = self._print_function_application(expr.application)
        return app
    
    def _print_signature_literal(self, expr: SignatureLiteral) -> str:
        method_ref = self._print_method_reference(expr.method_ref)
        param_types = [self._print_vm_type(pt) for pt in expr.parameter_types]
        return f"sig:{method_ref}({', '.join(param_types)})"
    
    # Helper methods for schema elements
    
    def _print_type_and_id(self, tai: TypeAndId) -> str:
        ty = self._print_cvl_type(tai.decl_type)
        return f"{ty} {tai.id}"
    
    def _print_function_application(self, app: FunctionApplication) -> str:
        args = [self.print_expression(arg) for arg in app.params]
        app_base = f"{app.host_contract}." if app.host_contract else ""
        app_base += app.name
        if app.annotation:
            app_base += f"@{app.annotation}"
        return f"{app_base}({', '.join(args)})"
    
    def _print_method_reference(self, ref: MethodReference) -> str:
        contract = self._maybe_contract(ref.contract)
        return f"{contract}{ref.method_name}"
    
    def _print_vm_type(self, vmt: VMType) -> str:
        r = self._print_cvl_type(vmt.base_type)
        if not vmt.location:
            return r
        else:
            return f"{r} {vmt.location}"
    
    def _print_cvl_type(self, cvl_type: CVLType) -> str:
        match cvl_type.type:
            case "mapping":
                return f"mapping({self._print_cvl_type(cvl_type.key_type)} => {self._print_cvl_type(cvl_type.value_type)})"
            case "dyn_array":
                return f"{self._print_cvl_type(cvl_type.base_type)}[]"
            case "static_array":
                return f"{self._print_cvl_type(cvl_type.base_type)}[{cvl_type.n}]"
            case "primitive":
                return cvl_type.type_name
            case "storage_type":
                return "storage"
            case "contract_type":
                return f"{cvl_type.host_contract}.{cvl_type.type_name}"
    
    # Command pretty printing
    
    def print_command(self, cmd: Command, ident: LineBuilder):
        """Print a command."""
        match cmd.type:
            case "declaration":
                ident.line(self._print_declaration_cmd(cmd))
            case "assignment":
                ident.line(self._print_assignment_cmd(cmd))
            case "havoc":
                ident.line(self._print_havoc_cmd(cmd))
            case "assume":
                ident.line(self._print_assume_cmd(cmd))
            case "assume_invariant":
                ident.line(self._print_assume_invariant_cmd(cmd))
            case "apply":
                ident.line(self._print_apply_cmd(cmd))
            case "reset_storage":
                ident.line(self._print_reset_storage_cmd(cmd))
            case "assert":
                ident.line(self._print_assert_cmd(cmd))
            case "satisfy":
                ident.line(self._print_satisfy_cmd(cmd))
            case "return":
                ident.line(self._print_return_cmd(cmd))
            case "revert":
                ident.line(self._print_revert_cmd(cmd))
            case "if":
                self._print_if_cmd(cmd, ident, False)
            case "block":
                self._print_block_cmd(cmd, ident)
    
    def _print_declaration_cmd(self, cmd: DeclarationCmd) -> str:
        var = self._print_type_and_id(cmd.variable)
        if cmd.initial_value:
            init_val = self.print_expression(cmd.initial_value)
            return f"{var} = {init_val};"
        else:
            return var + ";"
    
    def _print_assignment_cmd(self, cmd: AssignmentCmd) -> str:
        lhs = [self._print_left_hand_side(lhs) for lhs in cmd.left_hand_sides]
        rhs = self.print_expression(cmd.expression)
        if len(lhs) == 1:
            return f"{lhs[0]} = {rhs};"
        else:
            return f"({', '.join(lhs)}) = {rhs};"
    
    def _print_havoc_cmd(self, cmd: HavocCmd) -> str:
        targets = [self._print_left_hand_side(target) for target in cmd.targets]
        fragment = f"havoc {', '.join(targets)}"
        if cmd.assumption:
            fragment += f" assuming {self.print_expression(cmd.assumption)}"
        return fragment + ";"
    
    def _print_assume_cmd(self, cmd: AssumeCmd) -> str:
        expr = self.print_expression(cmd.expression)
        return f"require({expr}, \"{cmd.message}\");"
    
    def _print_assume_invariant_cmd(self, cmd: AssumeInvariantCmd) -> str:
        args = [self.print_expression(arg) for arg in cmd.arguments]
        fragment = f"requireInvariant {cmd.invariant_name}"
        if len(args) > 0:
            fragment += self.print_and_join(cmd.arguments, self.print_expression, prefix="(", postfix=")")
        return fragment + ";"
    
    def _print_apply_cmd(self, cmd: ApplyCmd) -> str:
        target = self._print_function_application(cmd.target)
        return target + ";"
    
    def _print_reset_storage_cmd(self, cmd: ResetStorageCmd) -> str:
        target = self.print_expression(cmd.target)
        return f"reset_storage {target};"
    
    def _print_assert_cmd(self, cmd: AssertCmd) -> str:
        expr = self.print_expression(cmd.expression)
        return f"assert({expr}, \"{cmd.message}\");"
    
    def _print_satisfy_cmd(self, cmd: SatisfyCmd) -> str:
        expr = self.print_expression(cmd.expression)
        return f"satisfy({expr}, \"{cmd.message}\");"
    
    def _print_return_cmd(self, cmd: ReturnCmd) -> str:
        values = [self.print_expression(val) for val in cmd.values]
        if len(values) == 0:
            return "return;"
        elif len(values) == 1:
            return f"return {values[0]};"
        else:
            return f"return ({', '.join(values)});"
    
    def _print_revert_cmd(self, cmd: RevertCmd) -> str:
        if cmd.message:
            return f"revert(\"{cmd.message}\");"
        else:
            return "revert;"
        
    def _print_elif(self, else_blk: ElseBlock, ident: LineBuilder):
        match else_blk.type:
            case "else":
                ident.line("} else {")
                with ident.indent() as nested:
                    for c in else_blk.commands:
                        self.print_command(c, nested)
                ident.line("}")
            case "if":
                self._print_if_cmd(else_blk, ident, True)


    
    def _print_if_cmd(self, cmd: IfCmd, ident: LineBuilder, is_elseif: bool):
        condition = self.print_expression(cmd.condition)
        if_line = f"if({condition}) {{"
        if is_elseif:
            if_line = "} else " + if_line
        ident.line(if_line)
        with ident.indent() as nested:
            for c in cmd.then_cmd:
                self.print_command(c, nested)
        if cmd.else_block:
            self._print_elif(cmd.else_block, ident)
        else:
            ident.line("}")
    
    def _print_block_cmd(self, cmd: BlockCmd, ident: LineBuilder):
        ident.line("{")
        with ident.indent() as nested:
            for c in cmd.commands:
                self.print_command(c, nested)
        ident.line("}")
    
    def _print_left_hand_side(self, lhs: LeftHandSide) -> str:
        match lhs.type:
            case "id":
                return lhs.name
            case "array_access":
                base = self._print_left_hand_side(lhs.base)
                index = self.print_expression(lhs.index)
                return f"{base}[{index}]"
            case "field_access":
                base = self._print_left_hand_side(lhs.base)
                return f"{base}.{lhs.field_name}"
    
    def _print_namedvm_param(self, param: NamedVMParam) -> str:
        ty = self._print_cvl_type(param.param_type)
        return f"{ty} {param.name}"
    
    # Top-level block printing
    
    def print_basic_block(self, block: BasicBlock) -> str:
        """Print a top-level basic block."""
        builder = LineBuilder()
        match block.type:
            case "rule":
                self._print_rule_block(block, builder)
            case "func_def":
                self._print_function_def(block, builder)
            case "invariant":
                self._print_invariant(block, builder)
            case "sort_decl":
                self._print_sort_def(block, builder)
            case "ghost_def":
                self._print_ghost_def(block, builder)
            case "macro_def":
                self._print_macro_def(block, builder)
            case "hook_def":
                self._print_hook_def(block, builder)
            case "methods_block":
                self._print_methods_block(block, builder)
        return "\n".join(builder.buffer)
    
    def _print_rule_block(self, rule: RuleBlock, ident: LineBuilder):
        params = self.print_and_join(rule.rule_params, self._print_type_and_id)
        with ident.line_builder() as lb:
            lb.append(f"rule {rule.rule_name}")
            if len(params) > 0:
                lb.append(f"({params})")
            if rule.filtered_block:
                lb.append(" filtered {")
                with ident.indent() as nested:
                    for ind, filt in enumerate(rule.filtered_block):
                        append_comma = ind != len(rule.filtered_block) - 1
                        self._print_filtered_block(filt, nested, append_comma)
                lb.append("}")
            lb.append(" {")
        with ident.indent() as nested:
            self._print_code_block(rule.block, nested)
        ident.line("}")
    
    def _print_function_def(self, func: FunctionDef, ident: LineBuilder):
        params = self.print_and_join(func.params, self._print_type_and_id)
        return_types = [self._print_cvl_type(t) for t in func.return_value]

        function_open = f"function {func.name}({params})"
        if len(return_types) == 0:
            function_open += " {"
        elif len(return_types) == 1:
            function_open += f" returns {return_types[0]} {{"
        else:
            function_open += f" returns ({', '.join(return_types)}) {{"
        ident.line(function_open)
        with ident.indent() as nested:
            self._print_code_block(func.block, nested)
        ident.line("}")
            
    def _print_invariant(self, inv: Invariant, ident: LineBuilder):
        params = [self._print_type_and_id(p) for p in inv.invariant_params]
        expr = self.print_expression(inv.invariant_expression)
        with ident.line_builder() as lb:
            lb.append(f"invariant {inv.name}")
            if params:
                lb.append(self.print_and_join(inv.invariant_params, self._print_type_and_id, prefix="(", postfix=")"))
            lb.append(f" {expr}")
            if inv.filter:
                lb.append(" filtered {")
                with ident.indent() as nested:
                    self._print_filtered_block(inv.filter, nested)
                lb.append("}")
            if inv.proofs:
                lb.append(" {")
            else:
                return
        with ident.indent() as nested:
            for proof in inv.proofs:
                self._print_proof_block(proof, nested)
        ident.line("}")
    
    def _print_sort_def(self, sort: SortDef, ident: LineBuilder):
        ident.line(f"sort {sort.sort_name};")
    
    def _print_ghost_def(self, ghost: GhostDef, ident: LineBuilder):
        with ident.line_builder() as lb:
            if ghost.persistent:
                lb.append("persistent ")
            lb.append(f"ghost ")
            ty = ghost.ghost_type
            match ty.type:
                case "ghost_fun":
                    lb.append(ghost.ghost_name)
                    lb.append("(")
                    lb.append(self.print_and_join(ty.params, self._print_cvl_type))
                    lb.append(") returns ")
                    lb.append(self._print_cvl_type(ty.result_type))
                case "ghost_type":
                    lb.append(self._print_cvl_type(ty.base_type))
                    lb.append(f" {ghost.ghost_name}")
            if not ghost.axioms:
                lb.append(";")
                return
            lb.append(" {")
        with ident.indent() as nested:
            for ax in ghost.axioms:
                self._print_ghost_axiom(ax, nested)
        ident.line("}")

    def _print_ghost_axiom(self, ax: GhostAxiom, ident: LineBuilder):
        line = ""
        if ax.initial:
            line += "init_state "
        exp = self.print_expression(ax.exp)
        line += f"axiom {exp};"
        ident.line(line)
    
    def _print_macro_def(self, macro: MacroDef, lb: LineBuilder):
        params = self.print_and_join(macro.params, self._print_type_and_id, prefix="(", postfix=")")
        result_type = self._print_cvl_type(macro.result_type)
        expr = self.print_expression(macro.exp)
        
        lb.line(
            f"definition {macro.id}{params} returns {result_type} = {expr};"
        )
    
    def _print_hook_def(self, hook: HookDef, ident: LineBuilder):
        self._print_hook_pattern(hook.pattern, ident)
        with ident.indent() as nested:
            self._print_code_block(hook.block, nested)
        ident.line("}")
    
    def _print_methods_block(self, methods: MethodsBlock, ident: LineBuilder):
        ident.line("methods {")
        with ident.indent() as nested:
            for m in methods.method_entries:
                self._print_method_entry(m, nested)
        ident.line("}")
    
    def _print_code_block(self, block: CodeBlock, ident: LineBuilder):
        for c in block.commands:
            self.print_command(c, ident)
    
    def _print_filtered_block(self, filtered: FilteredBlock, ident: LineBuilder, append_comma: bool = False):
        expr = self.print_expression(filtered.e)
        line = f"{filtered.method_param} -> {expr}"
        if append_comma:
            line += ","
        ident.line(line)

    def print_and_join(self, elems: list[T], mapper: Callable[[T], str], *, prefix = "", postfix = "", delim: str = ", ") -> str:
        l = delim.join([mapper(it) for it in elems])
        return prefix + l + postfix
    
    def _print_proof_block(self, proof: ProofBlock, ident: LineBuilder):
        target = proof.target
        with ident.line_builder() as lb:
            lb.append("preserved")
            match target.type:
                case "special":
                    match target.special_target:
                        case "constructor":
                            lb.append(" constructor()")
                        case "fallback":
                            lb.append("fallback()")
                case "generic":
                    pass
                case "specific_method":
                    lb.append(" ")
                    if target.sig.host_contract:
                        lb.append(target.sig.host_contract + ".")
                    sig = self.print_and_join(target.sig.params, self._print_namedvm_param, prefix=f"{target.sig.name}(", postfix=")")
                    lb.append(sig)
            if proof.with_binding:
                lb.append(f" with (env {proof.with_binding.id})")
            lb.append(" {")
        with ident.indent() as nested:
            self._print_code_block(proof.block, nested)
        ident.line("}")

    def _type_to_pattern_name(self, ty: str) -> str:
        return ty[0:1].upper() + ty[1:]

    def _print_load_pattern(self, patt: LoadPattern, lb: LineInterleaver):
        lb.append(self._type_to_pattern_name(patt.type) + " ")
        lb.append(self._print_namedvm_param(patt.value_param) + " ")
        lb.append(self._print_slot_pattern(patt.slot_pattern))

    def _print_store_pattern(self, patt: StorePattern, lb: LineInterleaver):
        lb.append(self._type_to_pattern_name(patt.type) + " ")
        lb.append(self._print_slot_pattern(patt.slot_pattern) + " ")
        lb.append(self._print_namedvm_param(patt.new_value_param))
        if patt.old_value_param:
            lb.append(" " + self._print_namedvm_param(patt.old_value_param))

    def _print_slot_pattern(self, slot: SlotPattern) -> str:
        match slot.type:
            case "array_access":
                return self._print_slot_pattern(slot.base_pattern) + "[INDEX " + self._print_namedvm_param(slot.index_param) + "]"
            case "named":
                return slot.name
            case "field_access":
                return self._print_slot_pattern(slot.base_pattern) + "." + slot.field_name
            case "map_access":
                return self._print_slot_pattern(slot.base_pattern) + "[KEY " + self._print_namedvm_param(slot.key_param) + "]"

    def _print_hook_pattern(self, pattern: HookPattern, ident: LineBuilder):
        with ident.line_builder() as lb:
            lb.append("hook ")
            match pattern.type:
                case "sstore":
                    as_sstore_patt : StorePattern = cast(StorePattern, pattern)
                    self._print_store_pattern(as_sstore_patt, lb)
                case "tstore":
                    pat : TstoreHook = pattern
                    as_tstore_patt : StorePattern = cast(StorePattern, pat)
                    self._print_store_pattern(as_tstore_patt, lb)
                case "sload":
                    self._print_load_pattern(cast(LoadPattern, pattern), lb)
                case "tload":
                    self._print_load_pattern(cast(LoadPattern, pattern), lb)
                case "create":
                    lb.append("CREATE (" + self._print_namedvm_param(pattern.param) + ")")
                case "opcode":
                    lb.append(pattern.opcode_name)
                    if pattern.input_params:
                        lb.append(self.print_and_join(pattern.input_params, self._print_namedvm_param, prefix="(", postfix=")"))
                    if pattern.output_param:
                        lb.append(" ")
                        lb.append(self._print_namedvm_param(pattern.output_param))
            lb.append(" {")

    def _print_vm_param(self, vm_param: VMParam) -> str:
        res = self._print_vm_type(vm_param.ty)
        if vm_param.name:
            res += " " + vm_param.name
        return res
    
    def _maybe_contract(self, contract: str | None) -> str:
        if contract:
            return contract + "."
        else:
            return ""

    def _print_unresolved_target(self, target: UnresolvedInTarget) -> str:
        match target.type:
            case "all_in_contract":
                return f"{target.contract_name}._"
            case "catch_all":
                return "_._"
            case "with_signature":
                params = self.print_and_join(target.params, self._print_vm_param)
                return f"_.{target.method_name}({params})"
            case "specific_method":
                res = self._maybe_contract(target.host_contract)
                res += target.name
                params = self.print_and_join(target.param_types, self._print_vm_param)
                return f"{res}({params})"
            
    def _to_cvl_bool(self, b: bool) -> str:
        return "true" if b else "false"

    def _print_method_pattern(self, patt: DispatchPattern, add_comma: bool) -> str:
        res: str
        match patt.type:
            case "all_in_contract":
                res = f"{patt.contract_name}._"
            case "matching_signature":
                res = self._maybe_contract(patt.host_contract)
                res += patt.method_name
                params = self.print_and_join(patt.params, self._print_vm_type)
                res += f"({params})"
        return res + ("," if add_comma else "")
    
    def _print_havoc_summary(self, havoc: HavocingSummary) -> str:
        return havoc.havoc_keyword.upper()

    def _print_summary(self, summ: CallSummary) -> str:
        match summ.type:
            case "always":
                return f"ALWAYS({self.print_expression(summ.expression)})"
            case "havocing":
                return self._print_havoc_summary(summ)
            case "keyword":
                return self._print_keyword_summary(summ)
            case "dispatcher":
                return self._print_dispatcher_summary(summ)
            case "expression":
                return self._print_expression_summ(summ)

    def _print_dispatcher_summary(self, summ: DispatcherSummary) -> str:
        return f"DISPATCHER(use_fallback={self._to_cvl_bool(summ.use_fallback)}, optimistic={self._to_cvl_bool(summ.optimistic)})"

    def _print_keyword_summary(self, summ: KeywordSummary) -> str:
        return summ.summary_keyword.upper()
    
    def _print_expression_summ(self, summ: ExpressionSummary) -> str:
        res = self.print_expression(summ.expression)
        if not summ.expect_clause:
            return res
        ec = summ.expect_clause
        ec_str: str
        match ec.type:
            case "type":
                ec_str = self.print_and_join(ec.expected_types, self._print_vm_type, prefix=" expect (", postfix=")")
            case "void":
                ec_str = " expect void"
        return res + ec_str


    def _print_method_signature(self, meth_sig: MethodSignature) -> str:
        res = self._maybe_contract(meth_sig.method_ref.contract)
        res += meth_sig.method_ref.method_name
        res += self.print_and_join(
            meth_sig.parameters,
            self._print_vm_param,
            prefix="(",
            postfix=")"
        )
        res += f" {meth_sig.visibility}"
        res += self.print_and_join(meth_sig.return_types, self._print_vm_param, prefix=" returns (", postfix=")")
        if meth_sig.post_flags:
            res += " " + " ".join(meth_sig.post_flags)
        return res
    
    def _print_method_entry(self, entry: MethodEntry, ident: LineBuilder):
        with ident.line_builder() as lb:
            match entry.type:
                case "unresolved_dynamic":
                    lb.append("unresolved external in ")
                    lb.append(self._print_unresolved_target(entry.method_ref))
                    summ = entry.summary
                    optimistic = self._to_cvl_bool(summ.optimistic)
                    use_fallback = self._to_cvl_bool(summ.use_fallback)
                    lb.append(f"=> DISPATCH(use_fallback={use_fallback}, optimistic={optimistic}) [")
                    
                    with ident.indent() as nested:
                        for i, elem in enumerate(summ.patterns):
                            add_comma = i != len(summ.patterns) - 1
                            nested.line(self._print_method_pattern(elem, add_comma))
                    lb.append("]")
                    if summ.default_summary:
                        lb.append(" default " + self._print_havoc_summary(summ.default_summary))
                    lb.append(";")
                case "catch_all":
                    lb.append(f"function {entry.contract_name}._ external => ")
                    lb.append(self._print_havoc_summary(entry.summary))
                    lb.append(";")
                case "imported_function":
                    lb.append("function ")
                    lb.append(self._print_method_signature(entry.signature))
                    if entry.with_env:
                        lb.append(f" with (env {entry.with_env})")
                    if entry.summary:
                        lb.append(" => ")
                        lb.append(self._print_summary(entry.summary))
                    lb.append(";")

    def print_contract_import(self, imp: ContractImport) -> str:
        return f"using {imp.contract_name} as {imp.as_name};"
    
    def print_spec_import(self, imp: ImportSpec) -> str:
        return f"import \"{imp.spec_file}\";"

def pretty_print(obj: CVLFile) -> str:
    printer = CVLPrettyPrinter()
    to_ret = []
    for imp in obj.import_specs:
        to_ret.append(printer.print_spec_import(imp))
    for imp in obj.import_contract:
        to_ret.append(printer.print_contract_import(imp))
    for bb in obj.blocks:
        to_ret.append(printer.print_basic_block(bb))
        to_ret.append("")
    return "\n".join(to_ret)
