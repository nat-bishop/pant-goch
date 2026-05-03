// "Aria | Continuity Program <aria.7@continuity.halcyon.com>" → "Aria | Continuity Program"
export function stripAddress(field: string): string {
  if (!field) return field;
  const lt = field.indexOf("<");
  return lt > 0 ? field.slice(0, lt).trim() : field.trim();
}
