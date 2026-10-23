#!/usr/bin/env python3
"""
Certora Prover CEX Extraction & Analysis Tool
==============================================

A complete command-line tool to:
1. Extract counterexamples from Certora Prover JSON output
2. Analyze them using Claude API
3. Generate human-readable reports

Usage:
    # Extract and display CEX
    python extract_and_analyze_cex.py extract /path/to/output transfer_succeeds
    
    # Extract and analyze with Claude
    python extract_and_analyze_cex.py analyze /path/to/output transfer_succeeds
    
    # Save analysis to file
    python extract_and_analyze_cex.py analyze /path/to/output transfer_succeeds --output analysis.md

Requirements:
    - ANTHROPIC_API_KEY environment variable set
    - Certora Prover output directory with treeView results
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Optional, Any
import re

# For Claude integration
try:
    from anthropic import Anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    print("Warning: anthropic package not installed. Install with: pip install anthropic")


class CertoraResultsParser:
    """Parse Certora Prover output structure"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.treeview_dir = self.output_dir / "Reports" / "treeView"
        self._validate_directory()
    
    def _validate_directory(self):
        """Ensure the output directory contains expected Certora results"""
        if not self.output_dir.exists():
            raise FileNotFoundError(f"Output directory not found: {self.output_dir}")
        
        if not self.treeview_dir.exists():
            raise FileNotFoundError(
                f"treeView directory not found at {self.treeview_dir}\n"
                f"Expected Certora output structure:\n"
                f"  {self.output_dir}/Reports/treeView/treeViewStatus_*.json"
            )
    
    def find_latest_treeview(self) -> Path:
        """Find the latest treeViewStatus_*.json file"""
        status_files = sorted(self.treeview_dir.glob("treeViewStatus_*.json"))
        if not status_files:
            raise FileNotFoundError(f"No treeViewStatus files found in {self.treeview_dir}")
        
        latest = status_files[-1]
        print(f"✓ Found latest treeview: {latest.name}")
        return latest
    
    def load_treeview_json(self, treeview_path: Path) -> dict:
        """Load and parse treeViewStatus JSON"""
        with open(treeview_path) as f:
            return json.load(f)
    
    def find_rule_in_tree(self, tree: dict, rule_name: str) -> Optional[dict]:
        """Search for a rule node in the tree structure"""
        def search_node(node):
            if node.get("name") == rule_name:
                return node
            for child in node.get("children", []):
                result = search_node(child)
                if result:
                    return result
            return None
        
        for rule in tree.get("rules", []):
            result = search_node(rule)
            if result:
                return result
        return None
    
    def list_all_rules(self, tree: dict) -> list:
        """Extract all rule names and their statuses"""
        rules = []
        
        def collect_rules(node, path=""):
            rule_info = {
                "name": node.get("name"),
                "status": node.get("status"),
                "type": node.get("nodeType"),
                "has_cex": bool(node.get("output"))
            }
            rules.append(rule_info)
            
            for child in node.get("children", []):
                collect_rules(child)
        
        for rule in tree.get("rules", []):
            collect_rules(rule)
        
        return rules


class CEXExtractor:
    """Extract counterexample data from Certora output"""
    
    def __init__(self, parser: CertoraResultsParser):
        self.parser = parser
    
    def extract_cex_json(self, rule_node: dict, treeview_dir: Path) -> Optional[dict]:
        """Extract CEX JSON data from a violated rule"""
        if rule_node.get("status") != "VIOLATED":
            return None
        
        if not rule_node.get("output"):
            return None
        
        cex_file = rule_node["output"][0]
        cex_path = treeview_dir / cex_file
        
        if not cex_path.exists():
            raise FileNotFoundError(f"CEX file not found: {cex_path}")
        
        with open(cex_path) as f:
            return json.load(f)
    
    def format_cex_tree(self, cex_data: dict, depth: int = 0) -> str:
        """Format CEX as a tree structure for human reading"""
        if "callTrace" not in cex_data:
            return json.dumps(cex_data, indent=2)
        
        def format_node(node, depth=0):
            indent = "  " * depth
            
            # Extract message
            message = node.get("message", {})
            text = message.get("text", "")
            
            # Replace placeholders with arguments
            for i, arg in enumerate(message.get("arguments", [])):
                placeholder = f"{{{i}}}"
                arg_value = arg.get("value", "???")
                # Truncate very long values
                if len(arg_value) > 100:
                    arg_value = arg_value[:97] + "..."
                text = text.replace(placeholder, arg_value)
            
            lines = [f"{indent}→ {text}"]
            
            # Skip noisy nodes that confuse analysis
            skip_nodes = {
                "Setup",
                "Global State",
                "Evaluate branch condition",
                "unknown loop source code"
            }
            
            for child in node.get("childrenList", []):
                child_text = child.get("message", {}).get("text", "")
                if child_text not in skip_nodes:
                    lines.append(format_node(child, depth + 1))
            
            return "\n".join(lines)
        
        return format_node(cex_data["callTrace"])
    
    def extract_from_rule_name(self, rule_name: str) -> tuple[dict, str]:
        """
        Complete extraction pipeline:
        1. Find latest treeview
        2. Load JSON
        3. Find rule
        4. Extract CEX
        5. Format as tree
        
        Returns: (cex_data, formatted_tree)
        """
        treeview_path = self.parser.find_latest_treeview()
        tree = self.parser.load_treeview_json(treeview_path)
        
        # Find rule
        rule_node = self.parser.find_rule_in_tree(tree, rule_name)
        if not rule_node:
            raise ValueError(f"Rule '{rule_name}' not found in treeview")
        
        status = rule_node.get("status")
        print(f"✓ Found rule: {rule_name} (status: {status})")
        
        # Extract CEX
        cex_data = self.extract_cex_json(rule_node, treeview_path.parent)
        if not cex_data:
            raise ValueError(f"No counterexample found for rule {rule_name} (status: {status})")
        
        print(f"✓ Extracted CEX from JSON")
        
        # Format as tree
        formatted_tree = self.format_cex_tree(cex_data)
        
        return cex_data, formatted_tree


class ClaudeAnalyzer:
    """Analyze CEX using Claude API"""
    
    def __init__(self, api_key: Optional[str] = None):
        if not CLAUDE_AVAILABLE:
            raise ImportError("anthropic package required. Install with: pip install anthropic")
        
        import os
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Set it with:\n"
                "  export ANTHROPIC_API_KEY='sk-...'\n"
                "or pass via argument: --api-key sk-..."
            )
        
        self.client = Anthropic(api_key=key)
        self.model = "claude-3-5-sonnet-20241022"
    
    def create_analysis_prompt(self, rule_name: str, cex_tree: str, spec_content: Optional[str] = None) -> str:
        """Create the prompt for Claude to analyze the CEX"""
        
        system_prompt = """You are an expert Solidity developer and formal verification specialist. 
Your task is to analyze Certora Prover counterexamples and explain:
1. What scenario the counterexample demonstrates
2. Why the specification/rule failed
3. Specific code changes needed to fix it
4. Alternative approaches if multiple solutions exist

Be precise and actionable. If uncertain, state your uncertainty clearly."""
        
        spec_section = ""
        if spec_content:
            spec_section = f"""
## Original Specification (CVL):
```cvl
{spec_content}
```
"""
        
        user_prompt = f"""## Rule: {rule_name}

## Counterexample Execution Trace:
```
{cex_tree}
```
{spec_section}

## Analysis Request:

Please analyze this counterexample and provide:

1. **Scenario**: Describe in natural language what inputs/state trigger this counterexample
2. **Root Cause**: Identify which line of code or logic caused the specification violation
3. **Potential Fixes**: Suggest 2-3 specific code changes to address this
4. **Considerations**: Are there any design trade-offs? Is the spec too strict?
5. **Confidence**: How confident are you in this analysis? Any uncertainties?

Format your response as a structured report with clear sections."""
        
        return system_prompt, user_prompt
    
    def analyze_cex(self, rule_name: str, cex_tree: str, spec_content: Optional[str] = None) -> str:
        """Send CEX to Claude and get analysis"""
        
        system_prompt, user_prompt = self.create_analysis_prompt(rule_name, cex_tree, spec_content)
        
        print(f"\n📤 Sending to Claude API ({self.model})...")
        print("⏳ Waiting for analysis...\n")
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )
        
        analysis = response.content[0].text
        print("✓ Received analysis from Claude\n")
        
        return analysis


class CLIFormatter:
    """Format output for terminal display"""
    
    @staticmethod
    def format_cex_report(rule_name: str, cex_tree: str) -> str:
        """Format CEX for display"""
        report = f"""
{'='*70}
COUNTEREXAMPLE FOR RULE: {rule_name}
{'='*70}

{cex_tree}

{'='*70}
"""
        return report
    
    @staticmethod
    def format_analysis_report(rule_name: str, analysis: str) -> str:
        """Format analysis for display"""
        report = f"""
{'='*70}
CLAUDE ANALYSIS FOR RULE: {rule_name}
{'='*70}

{analysis}

{'='*70}
"""
        return report
    
    @staticmethod
    def format_rules_list(rules: list) -> str:
        """Format list of all rules for display"""
        # Filter to show only rule nodes
        rule_nodes = [r for r in rules if r["type"] in ["ROOT", "CUSTOM_RULE"]]
        
        if not rule_nodes:
            return "No rules found in treeview"
        
        output = "\nAvailable Rules in Treeview:\n"
        output += f"{'Rule Name':<40} {'Status':<15} {'CEX Available':<15}\n"
        output += "-" * 70 + "\n"
        
        for rule in rule_nodes:
            status = rule["status"] or "UNKNOWN"
            cex_avail = "✓ Yes" if rule["has_cex"] else "✗ No"
            rule_name = (rule["name"] or "N/A")[:40]
            output += f"{rule_name:<40} {status:<15} {cex_avail:<15}\n"
        
        return output


def main():
    parser = argparse.ArgumentParser(
        description="Extract and analyze Certora Prover counterexamples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all rules in output
  python extract_and_analyze_cex.py list /path/to/output
  
  # Extract and display CEX
  python extract_and_analyze_cex.py extract /path/to/output transfer_succeeds
  
  # Extract and analyze with Claude
  python extract_and_analyze_cex.py analyze /path/to/output transfer_succeeds
  
  # Save analysis to file
  python extract_and_analyze_cex.py analyze /path/to/output transfer_succeeds \\
    --output analysis.md --api-key sk-...
  
  # With CVL spec context
  python extract_and_analyze_cex.py analyze /path/to/output transfer_succeeds \\
    --spec path/to/spec.spec
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # LIST command
    list_parser = subparsers.add_parser("list", help="List all rules in output")
    list_parser.add_argument("output_dir", help="Path to Certora output directory")
    
    # EXTRACT command
    extract_parser = subparsers.add_parser("extract", help="Extract CEX and display")
    extract_parser.add_argument("output_dir", help="Path to Certora output directory")
    extract_parser.add_argument("rule_name", help="Name of rule to analyze")
    extract_parser.add_argument("--output", help="Save to file instead of stdout")
    
    # ANALYZE command
    analyze_parser = subparsers.add_parser("analyze", help="Extract CEX and analyze with Claude")
    analyze_parser.add_argument("output_dir", help="Path to Certora output directory")
    analyze_parser.add_argument("rule_name", help="Name of rule to analyze")
    analyze_parser.add_argument("--output", help="Save analysis to file")
    analyze_parser.add_argument("--spec", help="Path to CVL specification file")
    analyze_parser.add_argument("--api-key", help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        # Initialize parser
        certora_parser = CertoraResultsParser(args.output_dir)
        
        if args.command == "list":
            treeview_path = certora_parser.find_latest_treeview()
            tree = certora_parser.load_treeview_json(treeview_path)
            rules = certora_parser.list_all_rules(tree)
            
            print(CLIFormatter.format_rules_list(rules))
        
        elif args.command == "extract":
            extractor = CEXExtractor(certora_parser)
            cex_data, cex_tree = extractor.extract_from_rule_name(args.rule_name)
            
            report = CLIFormatter.format_cex_report(args.rule_name, cex_tree)
            
            if args.output:
                with open(args.output, "w") as f:
                    f.write(report)
                print(f"✓ Saved to: {args.output}")
            else:
                print(report)
        
        elif args.command == "analyze":
            if not CLAUDE_AVAILABLE:
                print("❌ Error: Claude analysis requires 'anthropic' package")
                print("Install with: pip install anthropic")
                sys.exit(1)
            
            # Extract CEX
            extractor = CEXExtractor(certora_parser)
            cex_data, cex_tree = extractor.extract_from_rule_name(args.rule_name)
            
            # Load spec if provided
            spec_content = None
            if args.spec:
                spec_path = Path(args.spec)
                if not spec_path.exists():
                    print(f"❌ Spec file not found: {args.spec}")
                    sys.exit(1)
                spec_content = spec_path.read_text()
                print(f"✓ Loaded spec from: {args.spec}")
            
            # Analyze with Claude
            analyzer = ClaudeAnalyzer(api_key=args.api_key)
            analysis = analyzer.analyze_cex(args.rule_name, cex_tree, spec_content)
            
            report = CLIFormatter.format_analysis_report(args.rule_name, analysis)
            
            if args.output:
                with open(args.output, "w") as f:
                    f.write(report)
                print(f"✓ Saved analysis to: {args.output}")
            else:
                print(report)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
