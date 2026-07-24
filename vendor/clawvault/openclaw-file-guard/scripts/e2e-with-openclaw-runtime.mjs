// Last-mile end-to-end: drive the plugin through the REAL OpenClaw hook-runner,
// targeting a REAL running ClawVault at http://127.0.0.1:8766.
//
// This bypasses the only missing leg in the pipeline — a real LLM issuing a
// tool_call. Everything else on the path is real:
//   * the plugin binary we ship in dist/index.js
//   * OpenClaw's own hook-runner (initializeGlobalHookRunner + runBeforeToolCall)
//   * ClawVault's HTTP event ingest + dashboard feed
//
// Prereq: `clawvault start` is running. Exits non-zero on any mismatch.

import axios from "axios";
import { execSync } from "node:child_process";

// OpenClaw is installed globally; find the install root via npm.
function resolveOpenClawRoot() {
  if (process.env.OPENCLAW_ROOT) return process.env.OPENCLAW_ROOT;
  try {
    const prefix = execSync("npm root -g", { encoding: "utf-8" }).trim();
    if (prefix) return `${prefix}/openclaw`;
  } catch {}
  return "/home/cs/.npm-global/lib/node_modules/openclaw";
}
const openclawRoot = resolveOpenClawRoot();

const hookRunnerPath = await (async () => {
  // Look up the hashed filename dynamically so this keeps working across OC updates.
  const fs = await import("node:fs/promises");
  const dir = `${openclawRoot}/dist`;
  const files = await fs.readdir(dir);
  const runner = files.find((f) => f.startsWith("hook-runner-global-") && f.endsWith(".js"));
  const helpers = files.find((f) => f.startsWith("hooks.test-helpers-") && f.endsWith(".js"));
  if (!runner || !helpers) {
    throw new Error("could not find OpenClaw hook-runner or test-helpers");
  }
  return { runner: `${dir}/${runner}`, helpers: `${dir}/${helpers}` };
})();

const runnerMod = await import(hookRunnerPath.runner);
const helpersMod = await import(hookRunnerPath.helpers);

// Exports are minified single-letter aliases; map by body identity.
const initializeGlobalHookRunner =
  runnerMod.i ?? runnerMod.initializeGlobalHookRunner;
const getGlobalHookRunner = runnerMod.t ?? runnerMod.getGlobalHookRunner;
const resetGlobalHookRunner = runnerMod.a ?? runnerMod.resetGlobalHookRunner;
const createMockPluginRegistry =
  helpersMod.n ?? helpersMod.createMockPluginRegistry;

if (
  typeof initializeGlobalHookRunner !== "function" ||
  typeof getGlobalHookRunner !== "function" ||
  typeof createMockPluginRegistry !== "function"
) {
  console.error("FAIL: OpenClaw internal API shape changed");
  process.exit(2);
}

const CLAWVAULT = process.env.CLAWVAULT_URL ?? "http://127.0.0.1:8766";

async function snapshotPluginEvents() {
  const res = await axios.get(`${CLAWVAULT}/api/scan-history`, {
    params: { limit: 50 },
    timeout: 2000,
  });
  return res.data.filter((e) => e.source === "openclaw-file-guard");
}

async function registerPluginIntoRuntime(mode) {
  // Reset any prior runner state so repeated runs are hermetic.
  if (typeof resetGlobalHookRunner === "function") resetGlobalHookRunner();

  const logger = {
    info: (m) => console.log(`  [plugin.info] ${m}`),
    warn: (m) => console.log(`  [plugin.warn] ${m}`),
    error: (m) => console.log(`  [plugin.err ] ${m}`),
    debug: () => {},
  };

  const handlers = new Map();
  const api = {
    logger,
    config: {},
    pluginConfig: {
      clawvaultUrl: CLAWVAULT,
      mode,
      refreshIntervalSeconds: 60,
      requestTimeoutMs: 2000,
    },
    version: "e2e-1.0.0",
    on: (name, handler) => handlers.set(name, handler),
  };

  const register = (await import("../dist/index.js")).default;
  register(api); // synchronous per OpenClaw contract

  // Fire gateway_start the same way OpenClaw would
  await handlers.get("gateway_start")?.(undefined, {});
  // Let config-client fetch from ClawVault
  await new Promise((r) => setTimeout(r, 400));

  // Build a mock registry containing the plugin's before_tool_call hook,
  // then hand it to OpenClaw's real hook runner.
  const registry = createMockPluginRegistry([
    {
      pluginId: "openclaw-file-guard",
      hookName: "before_tool_call",
      handler: handlers.get("before_tool_call"),
    },
  ]);
  initializeGlobalHookRunner(registry);

  const runner = getGlobalHookRunner();
  if (!runner) throw new Error("hook runner is null after init");
  if (!runner.hasHooks("before_tool_call")) {
    throw new Error("registry did not register before_tool_call hook");
  }
  return runner;
}

async function fireToolCall(runner, toolName, params, ctx = {}) {
  // Call the EXACT same API OpenClaw uses when a tool is about to run.
  return runner.runBeforeToolCall({ toolName, params }, ctx);
}

function assertEq(actual, expected, label) {
  const ok = JSON.stringify(actual) === JSON.stringify(expected);
  if (!ok) {
    console.error(`FAIL ${label}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
    process.exit(1);
  }
  console.log(`  ✓ ${label}`);
}

function assertTruthy(actual, label) {
  if (!actual) {
    console.error(`FAIL ${label}: got ${JSON.stringify(actual)}`);
    process.exit(1);
  }
  console.log(`  ✓ ${label}`);
}

async function scenarioStrictBlocks() {
  console.log("\n── Scenario 1: strict mode + sensitive tool call → real block + dashboard event");
  const before = (await snapshotPluginEvents()).length;
  const runner = await registerPluginIntoRuntime("strict");
  const result = await fireToolCall(
    runner,
    "read",
    { path: "/home/user/.ssh/id_rsa" },
    { agentId: "e2e-agent" },
  );
  // OpenClaw's runner merges results; plugin returned {block:true, blockReason:"..."},
  // the runner wraps it with `params: ...` but must preserve block.
  assertEq(result?.block, true, "OpenClaw runBeforeToolCall returns block=true");
  assertTruthy(result?.blockReason?.includes("id_rsa"), "blockReason mentions the path");

  await new Promise((r) => setTimeout(r, 500));
  const after = await snapshotPluginEvents();
  assertEq(after.length, before + 1, "ClawVault dashboard picked up exactly one new plugin event");
  const ev = after[0];
  assertEq(ev.action, "block", "event.action === block");
  assertEq(ev.source, "openclaw-file-guard", "event.source tag is correct");
  assertEq(ev.threat_level, "high", "event.threat_level is high");
  console.log(`  → dashboard shows: [${ev.action}] ${ev.threat_level} ${ev.input_preview}`);
}

async function scenarioLogDoesNotBlock() {
  console.log("\n── Scenario 2: log mode + sensitive tool call → NOT blocked but still reported");
  const before = (await snapshotPluginEvents()).length;
  const runner = await registerPluginIntoRuntime("log");
  const result = await fireToolCall(
    runner,
    "bash",
    { command: "cat ~/.env.local" },
    {},
  );
  assertEq(result?.block, undefined, "block is not set in log mode");

  await new Promise((r) => setTimeout(r, 500));
  const after = await snapshotPluginEvents();
  assertEq(after.length, before + 1, "ClawVault dashboard got one event even without blocking");
  assertEq(after[0].action, "log", "event.action === log");
}

async function scenarioBenignTransparent() {
  console.log("\n── Scenario 3: strict mode + benign tool call → transparent, no dashboard event");
  const before = (await snapshotPluginEvents()).length;
  const runner = await registerPluginIntoRuntime("strict");
  const result = await fireToolCall(
    runner,
    "read",
    { path: "/repo/README.md" },
    {},
  );
  assertEq(result?.block, undefined, "benign path is not blocked");
  await new Promise((r) => setTimeout(r, 400));
  const after = await snapshotPluginEvents();
  assertEq(after.length, before, "no dashboard event for benign path");
}

async function scenarioBlockStopsChain() {
  console.log("\n── Scenario 4: OpenClaw's fail-closed policy — a second (benign) hook cannot override a block");
  const before = (await snapshotPluginEvents()).length;

  // Build a registry with TWO hooks: our plugin's real handler + a dummy "allow everything"
  // hook registered LATER. OpenClaw's runner must honor block=true and short-circuit.
  if (typeof resetGlobalHookRunner === "function") resetGlobalHookRunner();

  const logger = {
    info: () => {},
    warn: () => {},
    error: () => {},
    debug: () => {},
  };
  const handlers = new Map();
  const api = {
    logger,
    config: {},
    pluginConfig: {
      clawvaultUrl: CLAWVAULT,
      mode: "strict",
      refreshIntervalSeconds: 60,
      requestTimeoutMs: 2000,
    },
    version: "e2e-1.0.0",
    on: (name, handler) => handlers.set(name, handler),
  };
  const register = (await import("../dist/index.js")).default;
  register(api);
  await handlers.get("gateway_start")?.(undefined, {});
  await new Promise((r) => setTimeout(r, 400));

  const registry = createMockPluginRegistry([
    {
      pluginId: "openclaw-file-guard",
      hookName: "before_tool_call",
      handler: handlers.get("before_tool_call"),
    },
    {
      pluginId: "dummy-allow",
      hookName: "before_tool_call",
      handler: async () => ({ params: undefined }), // tries to un-block
    },
  ]);
  initializeGlobalHookRunner(registry);

  const runner = getGlobalHookRunner();
  const result = await runner.runBeforeToolCall(
    { toolName: "read", params: { path: "/home/user/.ssh/id_rsa" } },
    {},
  );
  assertEq(result?.block, true, "block survives even with a later permissive hook");

  await new Promise((r) => setTimeout(r, 500));
  const after = await snapshotPluginEvents();
  assertEq(after.length, before + 1, "still exactly one new event on the dashboard");
}

async function main() {
  // Sanity-check ClawVault is actually up
  try {
    await axios.get(`${CLAWVAULT}/api/health`, { timeout: 2000 });
  } catch {
    console.error(
      `FAIL: ClawVault not reachable at ${CLAWVAULT}. Start it with 'clawvault start' first.`,
    );
    process.exit(2);
  }
  console.log(`Running real-OpenClaw-runtime e2e against ClawVault ${CLAWVAULT}`);

  await scenarioStrictBlocks();
  await scenarioLogDoesNotBlock();
  await scenarioBenignTransparent();
  await scenarioBlockStopsChain();

  console.log("\n════════════════════════════════════════════════════════");
  console.log("  ALL 4 SCENARIOS PASSED via real OpenClaw hook-runner ✓");
  console.log("════════════════════════════════════════════════════════\n");
}

main().catch((err) => {
  console.error("\nE2E RUN ERROR:", err?.stack ?? err?.message ?? err);
  process.exit(2);
});
