# Contract Red-Flag Detector

A web application that uses AI to automatically analyze legal contracts and identify potentially unfair or unlawful terms. Perfect for lawyers, legal teams, or anyone who needs to quickly review contracts for problematic clauses.

## What It Does

This tool uploads a contract (PDF or text), analyzes it using a large language model, and highlights red flags - terms that might be unfair, unlawful, or problematic. It provides:

- **Detailed analysis** of each problematic clause
- **Verbatim quotes** from the contract as evidence
- **Legal explanations** of why each term might be problematic
- **Risk assessments** and severity ratings
- **Suggested revisions** for problematic clauses
- **Annotated PDF** with highlighted findings

## Why You Might Need This

- **Save time**: Automatically scan contracts instead of reading every line
- **Catch issues**: Find unfair terms you might miss during manual review
- **Consistent analysis**: Same criteria applied to every contract
- **Documentation**: Get detailed reports with quotes and explanations
- **Learning tool**: Understand what makes a clause problematic

## Quick Start with Docker

The easiest way to get started is using Docker - it handles all the dependencies automatically.

### Step 1: Install Docker

If you don't have Docker installed:

- **macOS/Windows**: Download [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- **Linux**: Install via your package manager (e.g., `sudo apt install docker.io docker-compose`)

### Step 2: Set Up Your Environment

1. Copy the example environment file:
   ```bash
   cp env.example .env
   ```

2. Open `.env` in any text editor and add your API key. You'll need at least one of these:
   - `NEBIUS_API_KEY=your_key_here` (for Nebius AI)
   - `OPENAI_API_KEY=your_key_here` (for OpenAI)

   That's the minimum you need to get started! See the [Environment Variables](#environment-variables) section below for all available options.

### Step 3: Start the Application

From the repository root directory, run:

```bash
docker compose up -d --build
```

This will:
- Build the Docker image (first time only, takes a few minutes)
- Start the server in the background
- Make it available at `http://localhost:8000`

### Step 4: Open the Web Interface

Open your browser and go to: **http://localhost:8000/**

You'll see a simple web interface where you can:
- Upload a contract (PDF or text file)
- Configure analysis settings
- View results with highlighted findings

### Step 5: Try It Out

Click the "Try example" button to test with a sample contract, or upload your own PDF.

### Stopping the Server

When you're done, stop the server with:

```bash
docker compose down
```

## Environment Variables

The `.env` file controls how the application behaves. Here's what each setting does:

### Required Settings

**API Keys** (you need at least one):
- `NEBIUS_API_KEY` - Your Nebius AI API key
- `OPENAI_API_KEY` - Your OpenAI API key
- `NEBIUS_BASE_URL` - Nebius API endpoint (default: `https://api.tokenfactory.nebius.com/v1/`)

### Authentication

- `ENABLE_AUTH=0` - Set to `1` to enable password protection (default: disabled)
- `BASIC_AUTH_USER=demo` - Username for login (default: `demo`)
- `BASIC_AUTH_PASS=demo` - Password for login (default: `demo`)

⚠️ **Important**: If you make your server accessible to others, change the default username and password!

### File Limits

- `MAX_PAGES=20` - Maximum number of pages to process (0 = unlimited)
- `MAX_PARAGRAPHS=0` - Maximum number of paragraphs (0 = unlimited, useful for debugging)
- `MAX_FILE_MB=15` - Maximum file size in megabytes
- `TTL_HOURS=24` - How long to keep job results before cleanup

### Model Configuration

- `OPENAI_MODEL_STAGE1=Qwen/Qwen3-30B-A3B-Thinking-2507` - Model for first stage (router)
- `OPENAI_MODEL_STAGE2=Qwen/Qwen3-30B-A3B-Thinking-2507` - Model for second stage (classifier)

These work with any OpenAI-compatible API. The default models are for Nebius TokenFactory.

### AI Behavior

- `LLM_TEMPERATURE=0.0` - Controls randomness (0.0 = deterministic, higher = more creative)
- `LLM_SEED=42` - Optional seed for reproducible results (comment out to disable)
- `MAX_LLM_CONCURRENCY=8` - How many AI requests to process simultaneously

### Classification Mode

- `SINGLE_CLASSIFIER=1` - Use single-stage classification (1 = enabled, 0 = two-stage mode)

**Two modes available:**
- **Two-stage mode** (default): Optimized for UK residential tenancy agreements. First stage quickly identifies potential issues, second stage does detailed analysis.
- **Single-stage mode** (legacy): One-step analysis, useful for general contracts and debugging.

You can also switch modes in the web UI using the "Use legacy single-step mode" checkbox.

### Advanced Settings

Most users won't need to change these, but they're available for fine-tuning:

**Stage 1 (Router) Settings:**
- `STAGE1_MAX_CATEGORIES=10` - Max categories to consider per paragraph
- `T_ROUTER=0.3` - Confidence threshold for category selection
- `ROUTER_TOPK_PROBS=15` - How many categories to evaluate
- `STAGE1_CONTEXT_PARAGRAPHS=5` - How many previous paragraphs to include as context
- `STAGE1_MAX_INPUT_CHARS=60000` - Maximum input size to avoid errors
- `STAGE1_OVERLAP_FRAC=0.15` - Overlap between text chunks (prevents cutting clauses)
- `STAGE1_WINDOW_TOKENS=1100` - Size of text windows for analysis
- `STAGE1_STRIDE_TOKENS=300` - Step size between windows

**Stage 2 (Classifier) Settings:**
- `STAGE2_CONTEXT_PARAGRAPHS=3` - Neighboring paragraphs to include
- `STAGE2_MAX_CONTEXT_CHARS=6000` - Maximum context size

**Few-Shot Examples:**
- `ENABLE_FEWSHOT=1` - Enable example-based learning (1 = enabled)
- `FEWSHOT_MAX_PER_CATEGORY=2` - Maximum examples per category
- `FEWSHOT_CONTEXT_WINDOW=240` - Context size for examples

**Network:**
- `HTTP_PROXY=` - Optional HTTP proxy (leave empty if not needed)
- `HTTPS_PROXY=` - Optional HTTPS proxy (leave empty if not needed)

## How to Use the Web Interface

1. **Upload a Contract**: Click "Choose File" and select a PDF or text file
2. **Configure Settings** (optional):
   - Choose your API provider and model
   - Select classification mode (two-stage or single-stage)
   - Enable debug mode to see detailed AI prompts
3. **Process**: Click "Analyze Contract" and wait for analysis to complete
4. **Review Results**: 
   - See summary of findings
   - View detailed analysis for each red flag
   - Download annotated PDF with highlighted clauses
   - Check debug traces to see what the AI was thinking

## Making Your Server Accessible to Others

If you want to share your server with colleagues or clients, you can use Cloudflare Tunnel (free, no account needed).

### Using Cloudflare Tunnel

1. **Install Cloudflare Tunnel:**
   - macOS: `brew install cloudflare/cloudflare/cloudflared`
   - Linux/Windows: Download from [Cloudflare's website](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/)

2. **Start your server** (if not already running):
   ```bash
   docker compose up -d
   ```

3. **Create a tunnel** in a new terminal:
   ```bash
   cloudflared tunnel --url http://localhost:8000
   ```

4. **Share the URL** that appears (looks like `https://random-name.trycloudflare.com`)

The URL is temporary and changes each time you restart the tunnel. No account or configuration needed!

## Editing Prompts and AI Behavior

If you want to customize how the AI analyzes contracts, see **[PROMPTS_README.md](PROMPTS_README.md)** for detailed instructions.

### Quick Guide

**To edit prompts** (the instructions that tell the AI what to look for):
1. Open `app/prompts.py` in any text editor
2. Look for functions with "EDIT HERE" comments
3. Edit the text inside the `return """..."""` blocks
4. Restart Docker: `docker compose restart`

**To edit few-shot examples** (example clauses that teach the AI):
1. Open `data/fewshot_uk_redflags.txt` in any text editor
2. Edit the example clauses and expected behaviors
3. Restart Docker: `docker compose restart`

**Preview your changes** without running the server:
```bash
python3 show_prompt.py
```

For more details, see **[PROMPTS_README.md](PROMPTS_README.md)**.

### Editing Legal References Whitelist

The system validates legal references to filter out non-existent or "hallucinated" laws. The whitelist of valid UK statutes is stored in **`app/validate_legal_refs.py`**.

**File location:** `app/validate_legal_refs.py`  
**What it contains:** A simple Python set (`VALID_UK_STATUTES`) with strings - just a list of valid UK statute names

**To add or remove legal references:**

1. Open `app/validate_legal_refs.py` in any text editor
2. Find the `VALID_UK_STATUTES` set (starts around line 24)
3. Add or remove statute names as needed. It's just a list of strings in curly braces `{}`. For example:
   ```python
   VALID_UK_STATUTES = {
       "Consumer Rights Act 2015",
       "Housing Act 2004",
       "Your New Act 2024",  # Add your new act here - just add a new line with quotes
       # ... other acts
   }
   ```
4. Save the file
5. Restart Docker: `docker compose restart`

**Important notes:** 
- The list is case-insensitive - "Consumer Rights Act 2015" will match "consumer rights act 2015"
- The system also supports pattern matching (e.g., "Act Name YYYY, sections X-Y"), so you don't need to list every possible variation
- Common law references like "UK common law" are also supported
- If you add a new act, make sure to include both the full name and any common abbreviations (e.g., "Consumer Rights Act 2015" and "CRA 2015")

**To regenerate the list automatically** (optional):
```bash
python3 scripts/update_uk_tenancy_statutes.py
```
This script generates a comprehensive list based on known UK housing/tenancy legislation. You can then copy the output into `app/validate_legal_refs.py`.

## Troubleshooting

### Docker Issues

**Port already in use:**
- Change the port in `docker-compose.yml` (look for `8000:8000` and change the first number)
- Or stop whatever is using port 8000: `lsof -ti:8000 | xargs kill`

**Build fails:**
- Make sure Docker is running
- Check you have enough disk space
- Try: `docker compose down` then `docker compose up -d --build`

**Container won't start:**
- Check logs: `docker compose logs web`
- Look for error messages about API keys or missing files

### API Key Issues

**"API key is required" error:**
- Make sure you set `NEBIUS_API_KEY` or `OPENAI_API_KEY` in `.env`
- Restart Docker after changing `.env`: `docker compose restart`

**API calls failing:**
- Verify your API key is valid and has credits/quota
- Check `NEBIUS_BASE_URL` is correct (if using Nebius)
- Try a different API provider

### Analysis Issues

**No findings found:**
- This might be correct - the contract might not have problematic terms
- Try enabling "Full LLM debug" to see what the AI is thinking
- Check that you're using the right classification mode

**Processing is slow:**
- Reduce `MAX_PAGES` to process smaller documents
- Lower `MAX_LLM_CONCURRENCY` if you're hitting rate limits
- Check your API provider's rate limits

**Results look wrong:**
- Enable debug mode to see the exact prompts sent to the AI
- Check that you're using the right model for your use case
- Review the prompts in `app/prompts.py` - they might need adjustment

## Running Without Docker

If you prefer to run directly on your machine:

1. **Install Python 3.11+** and dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment** (same as Docker):
   ```bash
   cp env.example .env
   # Edit .env with your API keys
   ```

3. **Create required directories:**
   ```bash
   mkdir -p data/uploads data/results configs
   ```

4. **Run the server:**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

**Note**: You may need system dependencies for PDF processing (e.g., `poppler-utils` on Linux). Docker is recommended to avoid these issues.

## Example Contract

The "Try example" button uses a sample contract. To update it with a different contract:

```bash
./scripts/refresh_example_contract.sh
```

## Getting Help

- **Prompt editing**: See `PROMPTS_README.md`
- **API issues**: Check your API provider's documentation
- **Docker issues**: Check Docker logs with `docker compose logs`
- **Analysis questions**: Enable debug mode to see what the AI is doing

## What's Next?

Once you have it running:
1. Try the example contract to see how it works
2. Upload your own contracts
3. Review the findings and explanations
4. Customize prompts if needed (see `PROMPTS_README.md`)
5. Share with your team using Cloudflare Tunnel

Happy contract analyzing! 🎉
