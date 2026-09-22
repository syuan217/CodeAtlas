export function formatAmount(cents: number): string {
  return (cents / 100).toFixed(2);
}

export function parseAmount(text: string): number {
  const n = Number(text);
  if (!Number.isFinite(n)) {
    throw new Error(`bad amount: ${text}`);
  }
  return Math.round(n * 100);
}
