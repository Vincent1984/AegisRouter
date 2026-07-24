import axios, { type AxiosInstance } from "axios";
import type {
  EffectiveRules,
  Logger,
  PluginRuntimeConfig,
} from "./types.js";

const DEFAULT_WATCH_PATHS = [
  ".ssh/**",
  ".aws/credentials",
  ".aws/config",
  ".gnupg/**",
  ".kube/config",
];
const DEFAULT_WATCH_PATTERNS = ["id_rsa*", "id_ed25519*", "*.pem", ".env*"];

export class ConfigClient {
  private rules: EffectiveRules;
  private timer: NodeJS.Timeout | null = null;
  private readonly http: AxiosInstance;

  constructor(
    private readonly config: PluginRuntimeConfig,
    private readonly logger: Logger,
  ) {
    this.http = axios.create({
      baseURL: config.clawvaultUrl.replace(/\/+$/, ""),
      timeout: config.requestTimeoutMs,
    });
    this.rules = {
      watchPaths: [...DEFAULT_WATCH_PATHS, ...config.extraPaths],
      watchPatterns: [...DEFAULT_WATCH_PATTERNS],
      extensions: [...config.extraExtensions],
    };
  }

  getRules(): EffectiveRules {
    return this.rules;
  }

  async refreshOnce(): Promise<boolean> {
    try {
      const res = await this.http.get("/api/config/file-monitor");
      const data = res.data ?? {};
      const watchPaths: string[] = Array.isArray(data.watch_paths)
        ? data.watch_paths.filter((s: unknown) => typeof s === "string")
        : [];
      const watchPatterns: string[] = Array.isArray(data.watch_patterns)
        ? data.watch_patterns.filter((s: unknown) => typeof s === "string")
        : [];
      this.rules = {
        watchPaths: [...watchPaths, ...this.config.extraPaths],
        watchPatterns,
        extensions: [...this.config.extraExtensions],
      };
      this.logger.debug(
        `config refreshed: ${watchPaths.length} paths, ${watchPatterns.length} patterns`,
      );
      return true;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      this.logger.warn(`config refresh failed, keeping previous rules: ${msg}`);
      return false;
    }
  }

  start(): void {
    void this.refreshOnce();
    if (this.timer) return;
    const intervalMs = Math.max(5_000, this.config.refreshIntervalSeconds * 1000);
    this.timer = setInterval(() => void this.refreshOnce(), intervalMs);
    if (typeof this.timer.unref === "function") this.timer.unref();
  }

  stop(): void {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
  }
}
