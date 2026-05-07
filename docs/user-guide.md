# User Guide: Chat Workflow

This guide is for end users who want to use the Chat Workflow library to generate structured evaluation criteria through natural conversation with an LLM.

## Quick Start

Get started in three simple steps:

```bash
# 1. Install
git clone <repo> && cd chat-workflow && make

# 2. Configure
# Edit config.json to set your provider + model
export OPENROUTER_API_KEY=your-key    # or OPENAI_API_KEY, etc.

# 3. Activate and run
source .venv/bin/activate
chat-workflow evaluation-criteria generate-reviewed-criteria --context "evaluating job offers"
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Git
- An API key from an LLM provider (OpenRouter, OpenAI, Anthropic, etc.)

### Step-by-Step Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/chat-workflow.git
   cd chat-workflow
   ```

2. **Set up the environment**:
   ```bash
   make
   ```
   This command creates a Python virtual environment and installs all dependencies.

3. **Activate the virtual environment**:
   ```bash
   source .venv/bin/activate
   ```

## Configuration

### API Keys

The library supports multiple LLM providers. Set your API key as an environment variable:

```bash
# For OpenRouter
export OPENROUTER_API_KEY=your-openrouter-key

# For OpenAI
export OPENAI_API_KEY=your-openai-key

# For Anthropic
export ANTHROPIC_API_KEY=your-anthropic-key
```

### Provider Configuration

Edit `config.json` in the project root to configure your preferred provider and model:

```json
{
  "llm": {
    "provider": "openrouter",
    "model": "openrouter/google/gemini-2.0-flash-lite-001",
    "temperature": 0.7,
    "max_retries": 3
  }
}
```

**Available providers**: `openrouter`, `openai`, `anthropic`, `cohere`, `groq`

**Temperature**: Controls randomness (0.0 = deterministic, 1.0 = creative)

## Using the CLI

The `chat-workflow` command provides an interactive conversation interface to generate evaluation criteria.

### Basic Usage

Start a conversation with a context:

```bash
chat-workflow evaluation-criteria generate-reviewed-criteria --context "choosing a laptop"
```

The LLM will guide you through a conversation to generate structured evaluation criteria for your decision.

### Command Options

| Option | Description | Default |
|--------|-------------|---------|
| `--context` | Initial context for the conversation | "" |
| `--max-turns` | Maximum number of conversation turns | 10 |
| `--max-refinements` | Maximum number of refinement rounds | 3 |

### Examples

**Interactive conversation**:
```bash
chat-workflow evaluation-criteria generate-reviewed-criteria --context "choosing a laptop"
```

**Customize conversation length**:
```bash
chat-workflow evaluation-criteria generate-reviewed-criteria --context "gift ideas" --max-turns 5
```

**Combine options**:
```bash
chat-workflow evaluation-criteria generate-reviewed-criteria --context "evaluating job offers" --max-turns 8 --max-refinements 2
```

The LLM will guide you through a conversation to generate structured evaluation criteria for your decision.

### Command Options

| Option | Description | Default |
|--------|-------------|---------|
| `--context` | Initial context for the conversation | "" |
| `--max-turns` | Maximum number of conversation turns | 10 |
| `--max-refinements` | Maximum number of refinement rounds | 3 |

### Examples

**Interactive conversation**:
```bash
chat-workflow evaluation-criteria generate-reviewed-criteria --context "choosing a laptop"
```

**Limit conversation length**:
```bash
chat-workflow evaluation-criteria generate-reviewed-criteria --context "gift ideas" --max-turns 5
```

**Customize refinement rounds**:
```bash
chat-workflow evaluation-criteria generate-reviewed-criteria --context "evaluating job offers" --max-refinements 2
```

## Conversation Flow

When you run `chat-workflow evaluation-criteria generate-reviewed-criteria`, here's what happens:

1. **Initialization**: The LLM introduces itself and explains the process
2. **Context gathering**: You provide initial context about your decision
3. **Criteria generation**: The LLM helps you identify evaluation criteria
4. **Refinement**: You can refine, add, or remove criteria
5. **Completion**: The conversation ends with structured criteria

### During the Conversation

- **Type your responses** when prompted
- **Press Enter** to submit each response
- **Press Ctrl+C** to cancel at any time
- **Answer questions** to help the LLM understand your needs

### Conversation Limits

- **Default**: 10 turns (back-and-forth exchanges)
- **Maximum**: 20 turns (configurable with `--max-turns`)
- **Minimum**: 1 turn

If you reach the turn limit, the conversation ends and any criteria generated so far are saved.

## Output Files

The generated criteria are displayed at the end of the conversation. You can copy and save them manually if needed:

```json
{
  "context": "choosing a laptop",
  "criteria": [
    {
      "name": "Budget",
      "description": "Maximum amount I'm willing to spend",
      "weight": 0.3
    },
    {
      "name": "Performance",
      "description": "Processing speed and multitasking capability",
      "weight": 0.25
    }
  ]
}
```

### File Structure

- **context**: Your original conversation context
- **criteria**: Array of evaluation criteria objects
- Each criterion includes:
  - `name`: Short name of the criterion
  - `description`: Detailed explanation
  - `weight`: Importance relative to other criteria (0.0 to 1.0)

## Troubleshooting

### Common Issues

**"API key error"**:
- Check that your API key environment variable is set
- Verify the key is valid and has sufficient credits
- Ensure you're using the correct provider name in `config.json`

**"Provider not supported"**:
- Check the provider name in `config.json`
- Ensure you have the required environment variable set
- Verify the provider is supported by the litellm library

**"Configuration file error"**:
- Ensure `config.json` exists in the project root
- Check that the JSON syntax is valid
- Verify all required fields are present

**"Conversation failed"**:
- The LLM may have encountered an error
- Try again with a simpler context
- Check your internet connection

### Debug Mode

Enable debug logging to see detailed LLM interactions:

```bash
CHAT_WORKFLOW_DEBUG=1 chat-workflow evaluation-criteria generate-reviewed-criteria --context "test"
```

This streams all LLM requests and responses to stderr with timing information.

## Next Steps

After generating evaluation criteria, you can:

1. **Review the JSON output** to understand your decision framework
2. **Use the criteria** to evaluate options systematically
3. **Modify the criteria** manually if needed
4. **Run again** with different contexts for other decisions

## Getting Help

- **Documentation**: See other guides in the `docs/` directory
- **Issues**: Report problems on the GitHub repository
- **Questions**: Check the README for contact information

---

**Note**: This guide covers end-user usage only. For information on:
- Building your own workflows: See `workflow-author-guide.md`
- Contributing to the library: See `contributor-guide.md`
- Example workflow details: See `example-evaluation-criteria.md`