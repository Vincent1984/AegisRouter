# openclaw-file-guard

An OpenClaw plugin that intercepts `before_tool_call` events and blocks (or logs) tool
invocations that target sensitive file paths. Detected events are posted to a running
ClawVault dashboard so they appear in the unified Events feed alongside proxy and
file-monitor events.

## Why

ClawVault's transparent proxy sees what eventually leaves the box as HTTP traffic, but it
cannot see *intent*: when an agent calls a file-reading tool like `read` or `exec cat
~/.ssh/id_rsa`, ClawVault only observes the content after the fact. This plugin adds a
second line of defense at the OpenClaw tool-call layer — using the same sensitive-path
configuration the user has already set in the ClawVault web UI.

If this plugin is not installed, ClawVault keeps working exactly as before.

## How it works

1. On gateway start, fetch `watch_paths` and `watch_patterns` from
   `GET /api/config/file-monitor` on the configured ClawVault instance.
2. Register a `before_tool_call` handler. On each tool call, inspect the parameters for
   file paths. If any path matches a sensitive glob or extension, either:
   - `log` mode: report the event, let the call proceed
   - `strict` mode: report the event, return `{ block: true, blockReason: "..." }`
3. Every detection is POSTed to `/api/events/external` so the ClawVault dashboard
   renders it with a `PLUGIN` tag.
4. Config is refreshed every `refreshIntervalSeconds` (default 30s). If ClawVault is
   unreachable, the plugin falls back to the last-known config (or the built-in
   defaults) and never blocks the tool-call pipeline on network errors.

## Configuration

See `openclaw.plugin.json` for the full schema. Minimal example:

```json
{
  "clawvaultUrl": "http://127.0.0.1:8766",
  "mode": "strict"
}
```

## Build & test

```bash
npm install
npm run build   # emits dist/index.js
npm test        # vitest unit tests
```

## Real OpenClaw smoke test

After starting ClawVault and the OpenClaw gateway, the full user-input route can be
verified with a normal prompt that causes the model to issue a read tool call while
still matching ClawVault's file-monitor rules:

```bash
echo 'PORT=8080' > /tmp/.env.demo
openclaw agent --local --agent main \
  --message "Read /tmp/.env.demo and tell me what port is configured. It's a demo config file I made for testing, just 1-2 lines." \
  --json
```

In `strict` mode, the tool call should be blocked before execution and the ClawVault
dashboard should show a plugin event like:

```text
[block] high [READ] /tmp/.env.demo (rule: .env*)
```
