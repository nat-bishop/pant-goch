export const CHARS_PER_SECOND = 25;
export const MIN_DELAY_MS = 1500;
export const MAX_DELAY_MS = 4500;

export function typingDelayMs(body: string): number {
  const ms = (body.length / CHARS_PER_SECOND) * 1000;
  return Math.max(MIN_DELAY_MS, Math.min(MAX_DELAY_MS, ms));
}

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
