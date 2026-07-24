import { describe, it, expect } from "vitest";
import os from "node:os";
import path from "node:path";
import {
  detect,
  extractCandidatePaths,
  matchPath,
} from "../src/path-detector.js";
import type { EffectiveRules } from "../src/types.js";

const RULES: EffectiveRules = {
  watchPaths: [".ssh/**", ".aws/credentials", ".env*"],
  watchPatterns: ["id_rsa*", "*.pem"],
  extensions: [".key", ".p12"],
};

describe("extractCandidatePaths", () => {
  it("pulls named path params", () => {
    const paths = extractCandidatePaths("read", { path: "/etc/hosts" });
    expect(paths).toContain("/etc/hosts");
  });

  it("expands ~ in paths", () => {
    const paths = extractCandidatePaths("read", { path: "~/.ssh/id_rsa" });
    expect(paths).toContain(path.join(os.homedir(), ".ssh/id_rsa"));
  });

  it("parses path-like tokens out of shell commands", () => {
    const paths = extractCandidatePaths("bash", {
      command: "cat ~/.ssh/id_rsa | base64",
    });
    expect(paths).toContain(path.join(os.homedir(), ".ssh/id_rsa"));
  });

  it("handles quoted shell args", () => {
    const paths = extractCandidatePaths("exec", {
      command: 'cat "/etc/shadow"',
    });
    expect(paths).toContain("/etc/shadow");
  });

  it("ignores flags and short tokens", () => {
    const paths = extractCandidatePaths("bash", { command: "ls -la" });
    expect(paths).not.toContain("-la");
  });
});

describe("matchPath", () => {
  it("hits basename glob", () => {
    expect(matchPath("/home/user/id_rsa", RULES)).toBe("id_rsa*");
  });

  it("hits directory glob", () => {
    expect(matchPath("/home/user/.ssh/authorized_keys", RULES)).toBe(".ssh/**");
  });

  it("hits extension rule", () => {
    expect(matchPath("/tmp/cert.key", RULES)).toBe("ext:.key");
  });

  it("hits dotenv variants", () => {
    expect(matchPath("/repo/.env.local", RULES)).toBe(".env*");
  });

  it("hits .pem anywhere", () => {
    expect(matchPath("/opt/certs/server.pem", RULES)).toBe("*.pem");
  });

  it("misses unrelated files", () => {
    expect(matchPath("/repo/README.md", RULES)).toBeNull();
  });

  it("misses tool output logs that happen to contain dot", () => {
    expect(matchPath("/repo/app.log", RULES)).toBeNull();
  });
});

describe("detect (end-to-end)", () => {
  it("returns first hit for sensitive path param", () => {
    const m = detect("read", { path: "~/.ssh/id_rsa" }, RULES);
    expect(m).not.toBeNull();
    // .ssh/** is earlier in the rules list, so it wins over id_rsa*
    expect(m?.matchedRule).toBe(".ssh/**");
    expect(m?.path).toBe(path.join(os.homedir(), ".ssh/id_rsa"));
  });

  it("falls back to id_rsa* when path is outside .ssh/", () => {
    const m = detect("read", { path: "/tmp/id_rsa_backup" }, RULES);
    expect(m?.matchedRule).toBe("id_rsa*");
  });

  it("returns hit via shell command", () => {
    const m = detect(
      "bash",
      { command: "cat /home/u/.aws/credentials" },
      RULES,
    );
    expect(m?.matchedRule).toBe(".aws/credentials");
  });

  it("returns null for benign tool call", () => {
    const m = detect("read", { path: "/repo/README.md" }, RULES);
    expect(m).toBeNull();
  });

  it("returns null when no path-like params exist", () => {
    const m = detect("search", { query: "password" }, RULES);
    expect(m).toBeNull();
  });
});
