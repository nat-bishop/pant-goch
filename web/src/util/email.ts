// "Janet Rivera <jrivera@stmaria-hospital.org>" → "Janet Rivera"
export function stripAddress(field: string): string {
  if (!field) return field;
  const lt = field.indexOf("<");
  return lt > 0 ? field.slice(0, lt).trim() : field.trim();
}
