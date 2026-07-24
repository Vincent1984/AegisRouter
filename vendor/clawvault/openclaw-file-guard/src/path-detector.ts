import * as path from "node:path";
import os from "node:os";
import picomatch from "picomatch";
import type { EffectiveRules, PathMatch } from "./types.js";

const CANDIDATE_PARAM_KEYS = [
  "path",
  "file",
  "file_path",
  "filename",
  "filepath",
  "src",
  "source",
  "target",
  "dst",
  "destination",
  "input_file",
  "output_file",
  "input",
  "output",
];

const SHELL_LIKE_TOOLS = new Set([
  "bash",
  "shell",
  "sh",
  "exec",
  "execute",
  "run_command",
  "run",
  "cmd",
  "terminal",
  "process",
  "computer",
]);

const FILE_READ_TOOLS = new Set([
  "read",
  "read_file",
  "open",
  "cat",
  "view",
  "view_file",
  "load",
  "fetch_file",
]);

const FILE_WRITE_TOOLS = new Set([
  "write",
  "write_file",
  "edit",
  "edit_file",
  "append",
  "save",
]);

function normalizeHome(p: string): string {
  if (p.startsWith("~/") || p === "~") {
    return path.join(os.homedir(), p.slice(1));
  }
  return p;
}

function looksLikePath(token: string): boolean {
  if (!token) return false;
  if (token.length < 2) return false;
  if (token.startsWith("-")) return false;
  return (
    token.startsWith("/") ||
    token.startsWith("~/") ||
    token.startsWith("./") ||
    token.startsWith("../") ||
    /\.[A-Za-z0-9]{1,8}$/.test(token) ||
    /\/[^\s]+/.test(token)
  );
}

function splitShellArgs(cmd: string): string[] {
  const tokens: string[] = [];
  const re = /"([^"]*)"|'([^']*)'|(\S+)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(cmd)) !== null) {
    tokens.push(m[1] ?? m[2] ?? m[3] ?? "");
  }
  return tokens;
}

export function extractCandidatePaths(
  toolName: string,
  params: Record<string, unknown>,
): string[] {
  const out = new Set<string>();
  const lowerTool = (toolName || "").toLowerCase();

  for (const key of CANDIDATE_PARAM_KEYS) {
    const v = params[key];
    if (typeof v === "string" && v.trim()) out.add(normalizeHome(v.trim()));
    if (Array.isArray(v)) {
      for (const item of v) {
        if (typeof item === "string" && item.trim()) {
          out.add(normalizeHome(item.trim()));
        }
      }
    }
  }

  if (SHELL_LIKE_TOOLS.has(lowerTool)) {
    const cmd =
      (params.command as string | undefined) ??
      (params.cmd as string | undefined) ??
      (params.script as string | undefined) ??
      "";
    if (typeof cmd === "string" && cmd) {
      for (const tok of splitShellArgs(cmd)) {
        if (looksLikePath(tok)) out.add(normalizeHome(tok));
      }
    }
  }

  if (FILE_READ_TOOLS.has(lowerTool) || FILE_WRITE_TOOLS.has(lowerTool)) {
    for (const v of Object.values(params)) {
      if (typeof v === "string" && looksLikePath(v)) {
        out.add(normalizeHome(v.trim()));
      }
    }
  }

  return [...out];
}

function basename(p: string): string {
  const idx = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"));
  return idx >= 0 ? p.slice(idx + 1) : p;
}

export function matchPath(
  candidate: string,
  rules: EffectiveRules,
): string | null {
  const lower = candidate.toLowerCase();

  for (const ext of rules.extensions) {
    if (!ext) continue;
    const e = ext.startsWith(".") ? ext.toLowerCase() : "." + ext.toLowerCase();
    if (lower.endsWith(e)) return `ext:${e}`;
  }

  const globs = [...rules.watchPaths, ...rules.watchPatterns];
  for (const g of globs) {
    if (!g) continue;
    const hasSlash = g.includes("/");
    const anchored = g.startsWith("/") || g.startsWith("**");
    const opts = { dot: true, basename: !hasSlash };
    const matchers = [picomatch(g, opts)];
    if (hasSlash && !anchored) {
      matchers.push(picomatch("**/" + g, opts));
    }
    const base = basename(candidate);
    for (const m of matchers) {
      if (m(candidate) || m(base)) return g;
    }
  }
  return null;
}

export function detect(
  toolName: string,
  params: Record<string, unknown>,
  rules: EffectiveRules,
): PathMatch | null {
  for (const candidate of extractCandidatePaths(toolName, params)) {
    const matched = matchPath(candidate, rules);
    if (matched) return { path: candidate, matchedRule: matched };
  }
  return null;
}
