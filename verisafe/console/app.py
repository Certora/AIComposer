"""
Textual debug console for exploring VFS and message history.
"""
from typing import Optional
from pathlib import Path

from textual.app import App, ComposeResult
from textual.widgets import Tree, Static, Header, Footer, TextArea, Button, ListView, ListItem, Label, TabbedContent, TabPane
from textual.containers import Horizontal, Vertical, VerticalScroll, Container
from textual.screen import ModalScreen
from rich.syntax import Syntax
from rich.text import Text
from rich.console import RenderableType, Group
from rich.markdown import Markdown

from verisafe.core.context import CryptoContext
from verisafe.core.state import CryptoStateGen
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage


class MessageListItem(ListItem):
    """Custom ListItem that stores message data."""
    
    def __init__(self, message: AnyMessage, index: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message_data = message
        self.message_index = index


class InterruptInputScreen(ModalScreen[str]):
    """Modal screen for interrupt mode text input."""
    
    CSS = """
    InterruptInputScreen {
        align: center middle;
        background: rgba(0, 0, 0, 0.8);
    }
    
    #input-container {
        width: 80%;
        height: 50%;
        border: solid $primary;
        background: $surface;
        padding: 1;
    }
    
    #input-area {
        height: 1fr;
        margin-bottom: 1;
    }
    
    #button-container {
        height: 3;
        align: center middle;
    }
    """
    
    def compose(self) -> ComposeResult:
        with Container(id="input-container"):
            yield Label("Enter explicit guidance:", id="input-label")
            yield TextArea(placeholder="Type your message here...", id="input-area")
            with Horizontal(id="button-container"):
                yield Button("Submit", variant="primary", id="submit-btn")
                yield Button("Cancel", variant="default", id="cancel-btn")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "submit-btn":
            text_area = self.query_one("#input-area", TextArea)
            self.dismiss(text_area.text)
        elif event.button.id == "cancel-btn":
            self.dismiss(None)


class DebugConsole(App):
    """Debug console for exploring VFS and message history."""
    
    CSS = """
    #header {
        height: 3;
    }
    
    #content {
        height: 1fr;
    }
    
    #sidebar {
        width: 50%;
        border-right: solid $primary;
    }
    
    #main-content {
        width: 50%;
        padding: 1;
    }
    
    #file-content-scroll, #message-content-scroll {
        height: 1fr;
        padding: 0;
    }
    
    #file-content, #message-content {
        padding: 1 2;
    }
    
    #interrupt-button {
        dock: bottom;
        height: 3;
        margin: 1;
    }
    
    Tree {
        padding: 1;
    }
    
    ListView {
        padding: 1;
    }
    """
    
    def __init__(
        self, 
        context: CryptoContext, 
        state: CryptoStateGen, 
        interrupt_mode: bool = False
    ):
        super().__init__()
        self.context = context
        self.state = state
        self.interrupt_mode = interrupt_mode
        
        # Build VFS structure
        self.vfs_files = {}
        for path, content_bytes in self.context.vfs_materializer.iterate(state):
            try:
                self.vfs_files[path] = content_bytes.decode('utf-8')
            except UnicodeDecodeError:
                self.vfs_files[path] = f"<Binary file: {len(content_bytes)} bytes>"
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=False, id="header")
        
        with Container(id="content"):
            with TabbedContent():
                with TabPane("VFS Explorer", id="vfs-tab"):
                    with Horizontal():
                        with Vertical(id="sidebar"):
                            yield Tree("Virtual File System", id="file_tree")
                        with Vertical(id="main-content"):
                            with VerticalScroll(id="file-content-scroll"):
                                yield Static("Select a file to view its contents", id="file-content")
                
                with TabPane("Message History", id="history-tab"):
                    with Horizontal():
                        with Vertical(id="sidebar"):
                            yield ListView(id="message-list")
                        with Vertical(id="main-content"):
                            with VerticalScroll(id="message-content-scroll"):
                                yield Static("Select a message to view its contents", id="message-content")
        
        if self.interrupt_mode:
            yield Button("Enter Guidance (Exits Console)", variant="primary", id="interrupt-button")
        
        yield Footer()
    
    def on_mount(self) -> None:
        # Populate VFS tree
        tree = self.query_one("#file_tree", Tree)
        tree.focus()
        self._populate_vfs_tree(tree)
        
        # Populate message history
        self._populate_message_history()
    
    def _populate_vfs_tree(self, tree: Tree) -> None:
        """Build hierarchical tree from VFS files."""
        tree.clear()
        root = tree.root
        root.expand()
        
        # Build tree structure
        nodes = {}  # path -> TreeNode
        
        for file_path in sorted(self.vfs_files.keys()):
            parts = Path(file_path).parts
            current = root
            current_path = ""
            
            for i, part in enumerate(parts):
                if current_path:
                    current_path = f"{current_path}/{part}"
                else:
                    current_path = part
                
                # Check if this node already exists
                if current_path not in nodes:
                    # Is this a file (last part) or directory?
                    is_file = (i == len(parts) - 1)
                    
                    if is_file:
                        # Detect file type for icon
                        ext = Path(part).suffix.lower()
                        if ext in ['.sol']:
                            icon = "⚡"
                        elif ext in ['.spec']:
                            icon = "📋"
                        elif ext in ['.md', '.txt']:
                            icon = "📝"
                        else:
                            icon = "📄"
                        label = f"{icon} {part}"
                    else:
                        label = f"📁 {part}"
                    
                    # Create node
                    node = current.add(label, data=current_path if is_file else None)
                    nodes[current_path] = node
                    
                    if not is_file:
                        node.expand()
                    
                    current = node
                else:
                    current = nodes[current_path]
    
    def _populate_message_history(self) -> None:
        """Populate the message history list."""
        message_list = self.query_one("#message-list", ListView)
        
        # Get messages from state
        messages = self.state['messages']
        
        for i, message in enumerate(messages):
            # Create message display
            if hasattr(message, 'type'):
                msg_type = getattr(message, 'type', 'unknown')
            else:
                msg_type = type(message).__name__
            
            # Use text() method for preview
            content = message.text()
            
            # Truncate long messages for list view
            preview = content[:100] + "..." if len(content) > 100 else content
            
            # Create list item with rich formatting
            item_text = Text()
            item_text.append(f"[{i+1:03d}] ", style="dim")
            item_text.append(f"{msg_type}: ", style="bold blue")
            item_text.append(preview)
            
            message_list.append(MessageListItem(message, i, Label(item_text)))
    
    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle VFS file selection."""
        node = event.node
        file_path = node.data
        
        if file_path and file_path in self.vfs_files:
            # Display file contents with syntax highlighting
            content = self.vfs_files[file_path]
            
            # Detect language from extension
            ext = Path(file_path).suffix.lower()
            language_map = {
                ".sol": "javascript",  # Close enough for syntax highlighting
                ".spec": "javascript",     # CVL is close to javascript
                ".md": "markdown",
                ".txt": "text",
            }
            language = language_map.get(ext, "text")
            syntax: RenderableType
            # Create syntax highlighted view
            try:
                syntax = Syntax(
                    content,
                    language,
                    theme="monokai",
                    line_numbers=True,
                    word_wrap=False,
                )
            except Exception:
                # Fallback to plain text if syntax highlighting fails
                syntax = Text(content)
            
            content_widget = self.query_one("#file-content", Static)
            content_widget.update(syntax)
    
    def _render_message_content(self, message: AnyMessage) -> RenderableType:
        """Render AnyMessage content with proper formatting."""
        # Get the text content using the guaranteed text() method
        text_content = message.text()

        # Also get raw content for better rendering of structured content
        raw_content = message.content
        
        # Handle structured content (list of thinking/text blocks)
        if isinstance(raw_content, list):
            return self._render_structured_content(raw_content, message)
        elif isinstance(raw_content, str):
            return self._format_text_content(raw_content)
        else:
            # Use the text() method result as fallback
            return self._format_text_content(text_content)
    
    def _format_text_content(self, content: str) -> RenderableType:
        """Format string content with markdown if appropriate."""
        try:
            # Check for HTML/XML tags - if present, don't render as markdown
            # as these are likely part of prompts or structured data
            import re
            html_tag_pattern = r'<[^>]+>'
            if re.search(html_tag_pattern, content):
                return Text(content)
            
            # Check if it looks like markdown
            markdown_markers = ['#', '*', '`', '```', '- ', '* ', '1. ', '## ', '### ']
            if any(marker in content for marker in markdown_markers):
                return Markdown(content)
            else:
                return Text(content)
        except Exception:
            return Text(content)
    
    def _get_content_label(self, message: AnyMessage, item_type: str) -> tuple[str, str]:
        """Get the appropriate label and style for content based on message type and item type."""
        if item_type == "thinking":
            return ("🤔 Thinking:", "bold yellow")
        elif item_type == "text" or item_type == "string":
            if isinstance(message, AIMessage):
                return ("🤖 AI Output:", "bold blue")
            elif isinstance(message, HumanMessage):
                return ("👤 Human Input:", "bold green")
            else:
                return ("💬 Response:", "bold blue")
        else:
            return (f"❓ Unknown Block ({item_type}):", "bold red")

    def _render_structured_content(self, raw_content: list, message: AnyMessage) -> RenderableType:
        """Render list content (thinking blocks, text blocks, tool use, etc.)."""
        renderables : list[RenderableType] = []
        content: RenderableType
        for i, item in enumerate(raw_content):
            if i > 0:
                # Add separator between items
                separator = Text("-" * 60, style="dim")
                renderables.append(separator)
            
            if isinstance(item, str):
                # Plain text block - check if it's markdown
                label, style = self._get_content_label(message, "string")
                header = Text(label, style=style)
                content = self._format_text_content(item)
                renderables.extend([header, content])
                
            elif isinstance(item, dict):
                item_type = item.get("type", "unknown")
                
                if item_type == "thinking":
                    # Thinking block - style similar to the web UI
                    label, style = self._get_content_label(message, "thinking")
                    header = Text(label, style=style)
                    thinking_text = item.get("thinking", "")
                    content = self._format_text_content(thinking_text)
                    renderables.extend([header, content])
                    
                elif item_type == "text":
                    # Regular text response - may contain markdown
                    label, style = self._get_content_label(message, "text")
                    header = Text(label, style=style)
                    text_content = item.get("text", "")
                    content = self._format_text_content(text_content)
                    renderables.extend([header, content])
                    
                elif item_type == "tool_use":
                    # Tool use block
                    header = Text("🔧 Tool Use:", style="bold green")
                    tool_info = Text()
                    tool_name = item.get("name", "unknown")
                    tool_id = item.get("id", "")
                    tool_info.append(f"Tool: {tool_name}\n", style="cyan")
                    if tool_id:
                        tool_info.append(f"ID: {tool_id}\n", style="dim")
                    
                    # Show input parameters if present
                    input_data = item.get("input", {})
                    if input_data:
                        tool_info.append("Parameters:\n", style="bold")
                        for key, value in input_data.items():
                            tool_info.append(f"  {key}: ", style="cyan")
                            if isinstance(value, str) and len(value) > 100:
                                tool_info.append(f"{value[:100]}...\n", style="dim")
                            else:
                                tool_info.append(f"{value}\n")
                    
                    renderables.extend([header, tool_info])
                    
                else:
                    # Unknown structured block
                    header = Text(f"❓ Unknown Block ({item_type}):", style="bold red")
                    details = Text()
                    for key, value in item.items():
                        details.append(f"  {key}: ", style="cyan")
                        if isinstance(value, str) and len(value) > 100:
                            details.append(f"{value[:100]}...\n", style="dim")
                        else:
                            details.append(f"{value}\n")
                    renderables.extend([header, details])
                    
            else:
                # Unknown item type
                header = Text(f"⚠️ Unknown Content ({type(item).__name__}):", style="bold red")
                content = Text(repr(item))
                renderables.extend([header, content])
        
        # Use Rich's Group to combine all renderables
        return Group(*renderables)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle message history selection."""
        if event.item and isinstance(event.item, MessageListItem):
            message: AnyMessage = event.item.message_data
            
            # Render message content with proper formatting
            rendered = self._render_message_content(message)
            
            # Update message content area
            content_widget = self.query_one("#message-content", Static)
            content_widget.update(rendered)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle interrupt button press."""
        if event.button.id == "interrupt-button":
            self.push_screen(InterruptInputScreen(), self._handle_interrupt_input)
    
    def _handle_interrupt_input(self, user_input: Optional[str]) -> None:
        """Handle the result from interrupt input."""
        if user_input:
            self.exit(user_input)
    
    def on_key(self, event) -> None:
        """Handle key presses for quick exit."""
        if event.key == "escape":
            self.exit(None)
        elif event.key == "ctrl+c":
            self.exit(None)


def debug_console(
    context: CryptoContext, 
    state: CryptoStateGen, 
    interrupt_mode: bool = False
) -> Optional[str]:
    """
    Launch the debug console.
    
    Args:
        context: The current CryptoContext
        state: The current CryptoStateGen state
        interrupt_mode: If True, enables interrupt mode with text input
    
    Returns:
        None if console was closed normally, or the user's input text if interrupt mode was used
    """
    console = DebugConsole(context, state, interrupt_mode)
    result = console.run()
    return result
