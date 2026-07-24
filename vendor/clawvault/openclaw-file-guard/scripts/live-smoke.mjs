// Live smoke test: hit real ClawVault + call built plugin's register() directly.
// Requires ClawVault already running on http://127.0.0.1:8766.

import axios from "axios";

const CLAWVAULT = process.env.CLAWVAULT_URL ?? "http://127.0.0.1:8766";

async function snapshotEvents() {
  const res = await axios.get(`${CLAWVAULT}/api/scan-history`, {
    params: { limit: 50 },
    timeout: 2000,
  });
  return res.data;
}

function makeApi(pluginCfg) {
  const handlers = new Map();
  const logs = [];
  return {
    api: {
      logger: {
        info: (m) => logs.push(["info", m]),
        warn: (m) => logs.push(["warn", m]),
        error: (m) => logs.push(["error", m]),
        debug: (m) => logs.push(["debug", m]),
      },
      config: {},
      pluginConfig: pluginCfg,
      version: "smoke-0.0.1",
      on: (event, handler) => handlers.set(event, handler),
    },
    handlers,
    logs,
  };
}

async function run() {
  const register = (await import("../dist/index.js")).default;

  const beforeCount = (await snapshotEvents()).length;
  console.log(`Events before: ${beforeCount}`);

  // Scenario 1: strict mode, sensitive path → expect block + event
  {
    const { api, handlers, logs } = makeApi({
      clawvaultUrl: CLAWVAULT,
      mode: "strict",
      refreshIntervalSeconds: 60,
      requestTimeoutMs: 2000,
    });
    await register(api);
    await handlers.get("gateway_start")(undefined, {});
    await new Promise((r) => setTimeout(r, 400));

    const result = await handlers.get("before_tool_call")(
      { toolName: "read", params: { path: "/home/u/.ssh/id_rsa" } },
      { agentId: "live-smoke-agent" },
    );
    console.log("S1 handler result:", JSON.stringify(result));

    await new Promise((r) => setTimeout(r, 500));
    const after = await snapshotEvents();
    const plugin = after.filter((e) => e.source === "openclaw-file-guard");
    console.log(`S1 plugin events now: ${plugin.length}`);
    if (plugin.length === 0) {
      console.error("S1 FAIL: no plugin event appeared on dashboard");
      console.error("logs:", logs);
      process.exit(1);
    }
    const ev = plugin[0];
    console.log(`S1 latest event: action=${ev.action} source=${ev.source} preview=${ev.input_preview}`);
    if (ev.action !== "block") {
      console.error("S1 FAIL: expected action=block");
      process.exit(1);
    }
  }

  // Scenario 2: log mode, sensitive path → expect no block but event
  {
    const { api, handlers } = makeApi({
      clawvaultUrl: CLAWVAULT,
      mode: "log",
      refreshIntervalSeconds: 60,
      requestTimeoutMs: 2000,
    });
    await register(api);
    await handlers.get("gateway_start")(undefined, {});
    await new Promise((r) => setTimeout(r, 400));

    const result = await handlers.get("before_tool_call")(
      { toolName: "bash", params: { command: "cat ~/.env.local" } },
      { agentId: "live-smoke-agent" },
    );
    console.log("S2 handler result:", JSON.stringify(result));
    if (result !== undefined) {
      console.error("S2 FAIL: expected undefined (no block)");
      process.exit(1);
    }
    await new Promise((r) => setTimeout(r, 500));
  }

  // Scenario 3: benign path → no block, no event
  {
    const { api, handlers } = makeApi({
      clawvaultUrl: CLAWVAULT,
      mode: "strict",
      refreshIntervalSeconds: 60,
      requestTimeoutMs: 2000,
    });
    await register(api);
    await handlers.get("gateway_start")(undefined, {});
    await new Promise((r) => setTimeout(r, 400));

    const before = (await snapshotEvents()).filter(
      (e) => e.source === "openclaw-file-guard",
    ).length;

    const result = await handlers.get("before_tool_call")(
      { toolName: "read", params: { path: "/repo/README.md" } },
      {},
    );
    console.log("S3 handler result:", JSON.stringify(result));
    if (result !== undefined) {
      console.error("S3 FAIL: benign path should not block");
      process.exit(1);
    }

    await new Promise((r) => setTimeout(r, 500));
    const after = (await snapshotEvents()).filter(
      (e) => e.source === "openclaw-file-guard",
    ).length;
    if (after !== before) {
      console.error(`S3 FAIL: benign path produced event (before=${before}, after=${after})`);
      process.exit(1);
    }
  }

  // Final state: print the last 5 plugin events
  const finalEvents = (await snapshotEvents()).filter(
    (e) => e.source === "openclaw-file-guard",
  );
  console.log(`\nFinal plugin events on dashboard: ${finalEvents.length}`);
  for (const e of finalEvents.slice(0, 5)) {
    console.log(`  [${e.action}] ${e.threat_level} ${e.input_preview}`);
  }
  console.log("\nAll live smoke scenarios PASSED ✓");
}

run().catch((err) => {
  console.error("SMOKE RUN ERROR:", err?.message ?? err);
  process.exit(2);
});
