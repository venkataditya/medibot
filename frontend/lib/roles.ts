export const ALL_COLLECTIONS = ["general", "clinical", "nursing", "billing", "equipment"] as const;

export function roleLabel(role: string): string {
  const words = role.replace(/_/g, " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}
