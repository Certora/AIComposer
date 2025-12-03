/**
 * CVL (Certora Verification Language) syntax definition for Prism
 * Converted from JFlex lexer
 * 
 * Usage:
 * 1. Populate the state variables (castFunctions, memoryLocations, etc.)
 * 2. Call buildCVLLanguage() to construct the language definition
 * 3. Use Prism.languages.cvl for highlighting
 * 
 * Order matters in Prism - patterns are matched in the order they appear in the object
 */

// State variables - populate these before calling buildCVLLanguage()
let castFunctions = new Set(["require_uint8", "require_int8", "assert_uint8", "assert_int8", "to_bytes1", "require_uint16", "require_int16", "assert_uint16", "assert_int16", "to_bytes2", "require_uint24", "require_int24", "assert_uint24", "assert_int24", "to_bytes3", "require_uint32", "require_int32", "assert_uint32", "assert_int32", "to_bytes4", "require_uint40", "require_int40", "assert_uint40", "assert_int40", "to_bytes5", "require_uint48", "require_int48", "assert_uint48", "assert_int48", "to_bytes6", "require_uint56", "require_int56", "assert_uint56", "assert_int56", "to_bytes7", "require_uint64", "require_int64", "assert_uint64", "assert_int64", "to_bytes8", "require_uint72", "require_int72", "assert_uint72", "assert_int72", "to_bytes9", "require_uint80", "require_int80", "assert_uint80", "assert_int80", "to_bytes10", "require_uint88", "require_int88", "assert_uint88", "assert_int88", "to_bytes11", "require_uint96", "require_int96", "assert_uint96", "assert_int96", "to_bytes12", "require_uint104", "require_int104", "assert_uint104", "assert_int104", "to_bytes13", "require_uint112", "require_int112", "assert_uint112", "assert_int112", "to_bytes14", "require_uint120", "require_int120", "assert_uint120", "assert_int120", "to_bytes15", "require_uint128", "require_int128", "assert_uint128", "assert_int128", "to_bytes16", "require_uint136", "require_int136", "assert_uint136", "assert_int136", "to_bytes17", "require_uint144", "require_int144", "assert_uint144", "assert_int144", "to_bytes18", "require_uint152", "require_int152", "assert_uint152", "assert_int152", "to_bytes19", "require_uint160", "require_int160", "assert_uint160", "assert_int160", "to_bytes20", "require_uint168", "require_int168", "assert_uint168", "assert_int168", "to_bytes21", "require_uint176", "require_int176", "assert_uint176", "assert_int176", "to_bytes22", "require_uint184", "require_int184", "assert_uint184", "assert_int184", "to_bytes23", "require_uint192", "require_int192", "assert_uint192", "assert_int192", "to_bytes24", "require_uint200", "require_int200", "assert_uint200", "assert_int200", "to_bytes25", "require_uint208", "require_int208", "assert_uint208", "assert_int208", "to_bytes26", "require_uint216", "require_int216", "assert_uint216", "assert_int216", "to_bytes27", "require_uint224", "require_int224", "assert_uint224", "assert_int224", "to_bytes28", "require_uint232", "require_int232", "assert_uint232", "assert_int232", "to_bytes29", "require_uint240", "require_int240", "assert_uint240", "assert_int240", "to_bytes30", "require_uint248", "require_int248", "assert_uint248", "assert_int248", "to_bytes31", "require_uint256", "require_int256", "assert_uint256", "assert_int256", "to_bytes32", "to_mathint", "require_address", "assert_address"]);
let memoryLocations = new Set(["calldata", "storage", "memory"]); // Set<string>
let hookableOpcodes = new Set(["SLOAD", "SSTORE", "CREATE", "TLOAD", "TSTORE", "ALL_SLOAD", "ALL_SSTORE", "ALL_TLOAD", "ALL_TSTORE", "ADDRESS", "BALANCE", "ORIGIN", "CALLER", "CALLVALUE", "CODESIZE", "CODECOPY", "GASPRICE", "EXTCODESIZE", "EXTCODECOPY", "EXTCODEHASH", "BLOCKHASH", "COINBASE", "TIMESTAMP", "NUMBER", "DIFFICULTY", "GASLIMIT", "CHAINID", "SELFBALANCE", "BASEFEE", "BLOBHASH", "BLOBBASEFEE", "MSIZE", "GAS", "LOG0", "LOG1", "LOG2", "LOG3", "LOG4", "CREATE1", "CREATE2", "CALL", "CALLCODE", "DELEGATECALL", "STATICCALL", "REVERT", "SELFDESTRUCT", "RETURNDATASIZE"]); // Set<string>
let preReturnMethodQualifiers = new Set(["external", "internal"]); // Set<string>
let postReturnMethodQualifiers = new Set(["optional", "envfree"]); // Set<string>
let constVals = new Set(["max_address", "max_uint", "max_uint104", "max_uint112", "max_uint120", "max_uint128", "max_uint136", "max_uint144", "max_uint152", "max_uint16", "max_uint160", "max_uint168", "max_uint176", "max_uint184", "max_uint192", "max_uint200", "max_uint208", "max_uint216", "max_uint224", "max_uint232", "max_uint24", "max_uint240", "max_uint248", "max_uint256", "max_uint32", "max_uint40", "max_uint48", "max_uint56", "max_uint64", "max_uint72", "max_uint8", "max_uint80", "max_uint88", "max_uint96"]); // Set<string>
let builtInFunctions = new Set(["keccak256", "ecrecover", "sha256"]); // Set<string>

/**
 * Helper to escape regex special characters in identifiers
 */
function escapeRegex(str) {
	return RegExp.escape(str);
}

/**
 * Build a regex pattern from a set of strings
 */
function buildPattern(items) {
	if (!items || items.size === 0) {
		return null;
	}
	// Sort by length (descending) to match longer identifiers first
	const sorted = Array.from(items).sort((a, b) => b.length - a.length);
	const escaped = sorted.map(escapeRegex);
	return new RegExp('\\b(?:' + escaped.join('|') + ')\\b');
}

/**
 * Builds the CVL language definition with current state variables
 * Call this after populating castFunctions, memoryLocations, etc.
 */
function buildCVLLanguage() {
	const lang = {};

	// Comments (highest priority - matched first)
	// Greedy ensures the entire comment is matched without backtracking
	lang['comment'] = [
		{
			pattern: /\/\/[^\n\r]*/,
			greedy: true
		},
		{
			pattern: /\/\*[\s\S]*?(?:\*\/|$)/,
			greedy: true
		}
	];

	// String literals
	// Greedy ensures the entire string is matched
	lang['string'] = {
		pattern: /"[^"]*"/,
		greedy: true
	};

	// Built-in CVL keywords (from the JFlex file)
	// Build keyword pattern dynamically to include pre/post return qualifiers
	const baseKeywords = ['sort', 'mapping', 'ghost', 'persistent', 'definition', 'axiom', 'hook', 
		'Sload', 'Sstore', 'Tload', 'Tstore', 'Create', 'ALWAYS', 'CONSTANT', 'PER_CALLEE_CONSTANT', 
		'NONDET', 'HAVOC_ECF', 'HAVOC_ALL', 'ASSERT_FALSE', 'AUTO', 'UNRESOLVED', 'ALL', 'DELETE', 
		'DISPATCHER', 'DISPATCH', 'default', 'norevert', 'withrevert', 'fallback', 'forall', 'exists', 
		'sum', 'usum', 'true', 'false', 'rule', 'unresolved', 'function', 'returns', 'expect', 'return', 
		'revert', 'havoc', 'assuming', 'require', 'requireInvariant', 'assert', 'satisfy', 'invariant', 
		'weak', 'strong', 'preserved', 'onTransactionBoundary', 'methods', 'description', 'good_description', 
		'filtered', 'reset_storage', 'if', 'else', 'as', 'using', 'import', 'use', 'builtin', 'override', 
		'xor', 'in', 'at', 'with', 'void', 'old', 'new', 'lastStorage', 'lastReverted', 'sig', 'STORAGE'];
	
	// Add pre-return and post-return method qualifiers to keywords
	const allKeywords = [...baseKeywords];
	if (preReturnMethodQualifiers.size > 0) {
		allKeywords.push(...preReturnMethodQualifiers);
	}
	if (postReturnMethodQualifiers.size > 0) {
		allKeywords.push(...postReturnMethodQualifiers);
	}

	if(memoryLocations.size > 0) {
		allKeywords.push(...memoryLocations);
	}
	
	// Sort by length (descending) and build pattern
	const sortedKeywords = allKeywords.sort((a, b) => b.length - a.length);
	const escapedKeywords = sortedKeywords.map(escapeRegex);
	lang['keyword'] = new RegExp('\\b(?:' + escapedKeywords.join('|') + ')\\b');

	// Dynamic patterns from state variables (in priority order based on JFlex)
	
	// Cast functions (Map, so we get keys)
	const castPattern = buildPattern(new Set(castFunctions.keys()));
	if (castPattern) {
		lang['builtin'] = castPattern;
	}

	// Hookable opcodes
	const opcodePattern = buildPattern(hookableOpcodes);
	if (opcodePattern) {
		if (lang['builtin']) {
			// If builtin already exists, make it an array
			lang['builtin'] = [lang['builtin'], opcodePattern];
		} else {
			lang['builtin'] = opcodePattern;
		}
	}

	// Built-in functions
	const bifPattern = buildPattern(builtInFunctions);
	if (bifPattern) {
		if (lang['function']) {
			if (Array.isArray(lang['function'])) {
				lang['function'].push(bifPattern);
			} else {
				lang['function'] = [lang['function'], bifPattern];
			}
		} else {
			lang['function'] = bifPattern;
		}
	}

	// Symbolic constants (max_uint256, etc.)
	const constPattern = buildPattern(constVals);
	if (constPattern) {
		lang['constant'] = constPattern;
	}

	// Operators (three-character must come before two-character, etc.)
	lang['operator'] = [
		/>>>/,   // BWRSHIFTWZEROS
		/>>=/,   // BWRSHIFTASSIGN
		/>>>=/,  // BWRSHIFTWZEROSASSIGN
		/<<=/,   // BWLSHIFTASSIGN
		/<=>/,   // IFF
		/<</,    // BWLSHIFT
		/>>/,    // BWRSHIFT
		/->/,    // MAPSTO
		/=>/,    // IMPLIES
		/\+\+/,  // PLUSPLUS
		/--/,    // MINUSMINUS
		/==/,    // EQ
		/!=/,    // NEQ
		/<=/,    // LEQ
		/>=/,    // GEQ
		/&&/,    // AND
		/\|\|/,  // OR
		/\+=/,   // PLUSASSIGN
		/-=/,    // MINUSASSIGN
		/\*=/,   // MULTASSIGN
		/\/=/,   // DIVASSIGN
		/%=/,    // MODASSIGN
		/&=/,    // BWANDASSIGN
		/\|=/,   // BWORASSIGN
		/\^=/,   // BWXORASSIGN
		/\+/,    // PLUS
		/-/,     // MINUS
		/\*/,    // MULT
		/\//,    // DIV
		/%/,     // MOD
		/!/,     // NOT
		/&/,     // BWAND
		/\|/,    // BWOR
		/\^/,    // BWXOR
		/~/,     // BWNOT
		/</,     // LT
		/>/,     // GT
		/=/,     // ASSIGN
		/@/      // AMPERSAT
	];

	// Numbers - hex must come before decimal
	lang['number'] = [
		/\b0x[0-9A-Fa-f]+\b/,
		/\b\d+\b/
	];

	// Types (hardcoded from rules.spec usage)
	const typePattern = /\b(?:uint256|address|bool|mathint|env|calldataarg|method)\b/;
	if (lang['builtin']) {
		if (Array.isArray(lang['builtin'])) {
			lang['builtin'].push(typePattern);
		} else {
			lang['builtin'] = [lang['builtin'], typePattern];
		}
	} else {
		lang['builtin'] = typePattern;
	}

	

	// Identifiers (matches JFlex Identifier pattern: [A-Za-z_$][A-Za-z_0-9$]*)
	lang['identifier'] = /\b[A-Za-z_$][A-Za-z_0-9$]*\b/;

	// Punctuation
	lang['punctuation'] = /[{}[\]().,;:?]/;

	return lang;
}

// Initialize with empty language definition
Prism.languages.cvl = buildCVLLanguage();

// Export for module systems and external configuration
if (typeof module !== 'undefined' && module.exports) {
	module.exports = {
		castFunctions,
		memoryLocations,
		hookableOpcodes,
		preReturnMethodQualifiers,
		postReturnMethodQualifiers,
		constVals,
		builtInFunctions,
		buildCVLLanguage
	};
}
