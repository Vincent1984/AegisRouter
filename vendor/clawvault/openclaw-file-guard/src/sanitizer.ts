import axios, { type AxiosInstance } from "axios";
import type { Logger, PluginRuntimeConfig } from "./types.js";

interface SanitizeResponse {
  success?: boolean;
  sanitized?: unknown;
  error?: unknown;
}

export class SanitizerClient {
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

  async sanitize(text: string): Promise<string | null> {
    try {
      const response = await this.http.post<SanitizeResponse>(
        "/api/openclaw/sanitize",
        { text },
      );
      const data = response.data;
      if (data?.success === true && typeof data.sanitized === "string") {
        return data.sanitized;
      }
      this.logger.warn("local sanitize failed: invalid ClawVault response");
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      this.logger.warn(`local sanitize failed: ${msg}`);
    }
    return null;
  }
}
