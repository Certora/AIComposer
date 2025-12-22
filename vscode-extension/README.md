# Certora AI Composer VS Code Extension

Intelligent, security-aware pair programmer for web3 developers. This extension integrates with the AI Composer backend to assist in smart contract development and formal verification.

## Features

- **Context Manager**: Easily manage the files (Interfaces, Specs, System Docs) that the AI uses for reasoning.
- **Agentic Chat**: Interactive sidebar for prompting the AI, viewing real-time reasoning logs, and responding to human-in-the-loop requests.
- **Prover Dashboard**: Dedicated view for monitoring formal verification history and viewing counterexample traces.
- **Custom Settings**: Beautiful, Cursor-like settings interface for managing backend connections and model preferences.

## Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) (v18 or higher)
- [npm](https://www.npmjs.com/)
- AI Composer backend server running locally.

### Installation

1. Navigate to the extension directory:
   ```bash
   cd vscode-extension
   ```
2. Install dependencies:
   ```bash
   npm install
   ```

### Development & Debugging

1. Open the `vscode-extension` folder in VS Code.
2. Compile the extension:
   ```bash
   npm run compile
   ```
3. Press `F5` to open a new **Extension Development Host** window with the extension loaded.
4. Ensure your AI Composer server is running (default port `8769`).
5. In the sidebar, click **Connect** to establish the WebSocket connection.

## Configuration

You can access the configuration by clicking the gear emoji (⚙️) in the extension sidebar header or the Chat view title bar.

Available settings:
- **Server Port**: The port the AI Composer WebSocket server is listening on (Default: `8769`).
- **AI Model**: Selection of Anthropic models (Default: `Claude 4 Sonnet`).
- **Memory Tool**: Enable/Disable memory tool access for the agent.
- **Debug Prompt Override**: Manually override the initial prompt for testing.

## License

MIT

