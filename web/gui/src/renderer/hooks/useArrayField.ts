import { useCallback, useMemo } from "react";

import { useConfigField } from "./useConfigField";

interface UseArrayFieldOptions<T> {
  path: string;
  createDefault: () => T;
  prepareClone?: (item: T) => T;
}

interface UseArrayFieldReturn<T> {
  items: T[];
  add: () => void;
  remove: (index: number) => void;
  clone: (index: number) => void;
}

export function useArrayField<T>({
  path,
  createDefault,
  prepareClone,
}: UseArrayFieldOptions<T>): UseArrayFieldReturn<T> {
  const [value, setValue] = useConfigField<T[]>(path);
  const items = useMemo(() => value ?? [], [value]);

  const add = useCallback(() => {
    setValue([...items, createDefault()]);
  }, [items, setValue, createDefault]);

  const remove = useCallback(
    (index: number) => {
      setValue(items.filter((_, i) => i !== index));
    },
    [items, setValue],
  );

  const clone = useCallback(
    (index: number) => {
      const cloned = JSON.parse(JSON.stringify(items[index])) as T;
      const prepared = prepareClone ? prepareClone(cloned) : cloned;
      const next = [...items];
      next.splice(index + 1, 0, prepared);
      setValue(next);
    },
    [items, setValue, prepareClone],
  );

  return { items, add, remove, clone };
}
