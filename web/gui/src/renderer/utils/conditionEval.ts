import type { Condition } from "@/types/uiSchema";

export function evaluateCondition(
  condition: Condition | undefined,
  config: Record<string, unknown>,
  predicates: Record<string, string[]>,
): boolean {
  if (!condition) return true;

  if ("eq" in condition) {
    return getField(config, condition.field) === condition.eq;
  }
  if ("neq" in condition) {
    return getField(config, condition.field) !== condition.neq;
  }
  if ("in" in condition) {
    return condition.in.includes(String(getField(config, condition.field) ?? ""));
  }
  if ("notIn" in condition) {
    return !condition.notIn.includes(String(getField(config, condition.field) ?? ""));
  }
  if ("predicate" in condition) {
    const group = predicates[condition.predicate];
    if (!group) return false;
    return group.includes(String(getField(config, "model_type") ?? ""));
  }
  if ("and" in condition) {
    return condition.and.every((c) => evaluateCondition(c, config, predicates));
  }
  if ("or" in condition) {
    return condition.or.some((c) => evaluateCondition(c, config, predicates));
  }
  if ("not" in condition) {
    return !evaluateCondition(condition.not, config, predicates);
  }

  return true;
}

function getField(config: Record<string, unknown>, path: string): unknown {
  const keys = path.split(".");
  let current: unknown = config;
  for (const key of keys) {
    if (current === null || current === undefined || typeof current !== "object") {
      return undefined;
    }
    current = (current as Record<string, unknown>)[key];
  }
  return current;
}
