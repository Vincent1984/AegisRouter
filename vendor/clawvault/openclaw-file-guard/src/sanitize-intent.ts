export type SanitizeIntent =
  | { action: "none" }
  | { action: "usage" }
  | { action: "sanitize"; text: string };

const SANITIZE_PATTERNS = [
  /^(?:\u8bf7)?(?:\u5e2e\u6211)?(?:\u8131\u654f\u4fe1\u606f|\u654f\u611f\u4fe1\u606f\u8131\u654f|\u4fe1\u606f\u8131\u654f|\u8131\u654f)[\uff1a:\s]+(?<text>[\s\S]+)$/u,
  /^(?:sanitize|redact|mask)[\uff1a:\s]+(?<text>[\s\S]+)$/iu,
];

const SANITIZE_USAGE_TERMS = new Set([
  "\u8131\u654f",
  "\u8131\u654f\u4fe1\u606f",
  "\u654f\u611f\u4fe1\u606f\u8131\u654f",
  "\u4fe1\u606f\u8131\u654f",
  "sanitize",
  "redact",
  "mask",
]);

const SANITIZE_QUESTION_RE =
  /(?:\u4ec0\u4e48\u662f|\u662f\u4ec0\u4e48\u610f\u601d|what is|explain|\u4ecb\u7ecd).*(?:\u8131\u654f|sanitize|redact|mask)/iu;

function stripTimestampPrefix(text: string): string {
  return text.replace(/^\[[^\]]{1,80}\]\s*/u, "");
}

export function parseSanitizeIntent(prompt: string): SanitizeIntent {
  const text = stripTimestampPrefix(prompt.trim());
  if (!text) return { action: "none" };

  const mentionPrefix = "@clawvault";
  if (!text.toLowerCase().startsWith(mentionPrefix)) return { action: "none" };

  const normalized = text.slice(mentionPrefix.length).trim();
  if (!normalized) return { action: "none" };
  if (SANITIZE_QUESTION_RE.test(normalized)) return { action: "none" };
  if (SANITIZE_USAGE_TERMS.has(normalized.toLowerCase())) {
    return { action: "usage" };
  }

  for (const pattern of SANITIZE_PATTERNS) {
    const match = pattern.exec(normalized);
    const payload = match?.groups?.text?.trim();
    if (payload) return { action: "sanitize", text: payload };
  }

  return { action: "none" };
}
