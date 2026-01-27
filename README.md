# Contract Red-Flag Detector (Demo)

## Preview Prompts (Without Docker)

To see what prompt will be sent to the LLM:

```bash
python3 show_prompt.py
```

This is useful for legal experts editing prompts - you can preview changes without running Docker.

See `PROMPTS_README.md` for detailed instructions on editing prompts.

## Installation

### Option 1: Run with Docker (Recommended)

This is the easiest way - Docker handles all dependencies automatically.

### 1) Copy the environment file

```bash
cp env.example .env
```

### 2) Edit `.env`

Set at least one API key:

- `NEBIUS_API_KEY=...` **or** `OPENAI_API_KEY=...`

Optional:

- `ENABLE_AUTH=1` to enable Basic Auth (default: enabled)
- `BASIC_AUTH_USER=your_username` (default: `demo`)
- `BASIC_AUTH_PASS=your_password` (default: `demo`)
- `LLM_TEMPERATURE=0.0` - Temperature for LLM calls (default: `0.0`, range: 0.0-2.0). Lower values make output more deterministic. Increase to `0.1` or `0.2` if you notice findings are missing.
- `MAX_FINDINGS_PER_CHUNK=5` - Maximum number of findings per text chunk (default: `5`). Increase if you notice findings being truncated.

**Note:** Basic Auth is enabled by default. When sharing your server publicly, change the default username/password!

### 3) Build and start

```bash
docker compose up -d --build
```

### 4) Open the app

Visit: `http://localhost:8000/`

### 5) Stop

```bash
docker compose down
```

### Option 2: Run Locally (Without Docker)

If you prefer to run directly on your machine:

1. **Install Python 3.11+** (if not already installed)

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment:**
   ```bash
   cp env.example .env
   # Edit .env and set your API keys
   ```

4. **Create necessary directories:**
   ```bash
   mkdir -p data/uploads data/results configs
   ```

5. **Run the server:**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

6. **Open the app:**
   Visit: `http://localhost:8000/`

**Note:** Make sure you have all system dependencies (like libraries for PDF processing) installed. Docker is recommended to avoid dependency issues.

## Make Server Accessible to Others

To share your local server with others over the internet (while it's running on your computer), use one of these free tunneling services:

1. **Install Cloudflare Tunnel:**
   ```bash
   # macOS
   brew install cloudflare/cloudflare/cloudflared
   
   # Or download from: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/
   ```

2. **Start your Docker server:**
   ```bash
   docker compose up -d
   ```

3. **Create a tunnel:**
   ```bash
   cloudflared tunnel --url http://localhost:8000
   ```
   
   You will get:
   2026-01-27T21:43:54Z INF Requesting new quick Tunnel on trycloudflare.com...
   2026-01-27T21:43:57Z INF +--------------------------------------------------------------------------------------------+
   2026-01-27T21:43:57Z INF |  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):  |
   2026-01-27T21:43:57Z INF |  https://carrier-oops-myrtle-convention.trycloudflare.com                                  |
   2026-01-27T21:43:57Z INF +--------------------------------------------------------------------------------------------+
   2026-01-27T21:43:57Z INF Cannot determine default configuration path. No file [config.yml config.yaml] in [~/.cloudflared ~/.cloudflare-warp ~/cloudflare-warp /etc/cloudflared /usr/local/etc/cloudflared]

   2026-01-27T21:43:57Z INF |  https://carrier-oops-myrtle-convention.trycloudflare.com      <- is URL (always different!)
   
5. **Share the URL** with others. They can access your server through this link!

6. **To stop:** Press `Ctrl+C` in the terminal where cloudflared is running.
