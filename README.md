# mac-action-orchestrator

A FastAPI app to orchestrate macOS actions (keyboard/mouse shortcuts) and run action batches from a web UI.

## Prerequisites
- macOS with UI automation permissions granted to the terminal/python process
- Python 3.10+

## Setup
1) Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

2) Install dependencies:
```bash
pip install -r requirements.txt
```

3) Create config and API key files:
```bash
cp config.example.json config.json
```
Edit `config.json` with your host/user/SSH options.

If you want to use Gemini planning, add keys to `apikeys.txt` (one per line). Example:
```text
yourkeys
```

## Run
Start the server:
```bash
uvicorn app:app --reload --port 8000
```
Open your browser at:
```text
http://127.0.0.1:8000
```

## Notes
- `config.json` is ignored in public repos. Keep real host/user/SSH details local.
- `apikeys.txt` should contain only your own keys and must not be committed.
