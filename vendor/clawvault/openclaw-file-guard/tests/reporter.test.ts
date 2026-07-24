import { describe, it, expect, vi } from "vitest";
import { Reporter } from "../src/reporter.js";
import type { ExternalEvent, Logger, PluginRuntimeConfig } from "../src/types.js";

const CFG: PluginRuntimeConfig = {
  clawvaultUrl: "http://127.0.0.1:1",
  mode: "log",
  localSanitize: true,
  extraPaths: [],
  extraExtensions: [],
  refreshIntervalSeconds: 30,
  requestTimeoutMs: 100,
};

const EVENT: ExternalEvent = {
  source: "openclaw-file-guard",
  category: "file_access_logged",
  threat_level: "medium",
  action: "log",
  tool_name: "read",
  file_path: "/home/u/.ssh/id_rsa",
  matched_rule: ".ssh/**",
  message: "test",
  risk_score: 6.0,
};

function makeLogger(): Logger & {
  warn: ReturnType<typeof vi.fn>;
  info: ReturnType<typeof vi.fn>;
  error: ReturnType<typeof vi.fn>;
  debug: ReturnType<typeof vi.fn>;
} {
  return {
    warn: vi.fn(),
    info: vi.fn(),
    error: vi.fn(),
    debug: vi.fn(),
  };
}

describe("Reporter", () => {
  it("does not throw when ClawVault is unreachable, logs a warning", async () => {
    const logger = makeLogger();
    const reporter = new Reporter(CFG, logger);

    await expect(reporter.report(EVENT)).resolves.toBeUndefined();
    expect(logger.warn).toHaveBeenCalled();
  });
});
