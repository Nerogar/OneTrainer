import { useCallback } from "react";

import { getByPath, useConfigStore } from "@/store/configStore";

export function useConfigField<T>(path: string | null | undefined): [T | undefined, (value: T) => void] {
  const value = useConfigStore((state) => {
    if (!path || state.config === null) return undefined;
    return getByPath(state.config as unknown as Record<string, unknown>, path) as T | undefined;
  });

  const updateField = useConfigStore((state) => state.updateField);

  const setValue = useCallback(
    (newValue: T) => {
      if (path) updateField(path, newValue);
    },
    [updateField, path],
  );

  return [value, setValue];
}
