import axios, { type AxiosInstance } from "axios";
import type { ExternalEvent, Logger, PluginRuntimeConfig } from "./types.js";

export class Reporter {
  private readonly http: AxiosInstance;

  constructor(
    private readonly config: PluginRuntimeConfig,
    private readonly logger: Logger,
  ) {
    this.http = axios.create({
      baseURL: config.clawvaultUrl.replace(/\/+$/, ""),
      timeout: config.requestTimeoutMs,
    });
  }

  async report(event: ExternalEvent): Promise<void> {
    try {
      await this.http.post("/api/events/external", event);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      this.logger.warn(`report to ClawVault failed: ${msg}`);
    }
  }
}
