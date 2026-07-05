import { PathPicker, type PathPickerProps } from "./PathPicker";

export type DirPickerProps = Omit<PathPickerProps, "mode" | "filters">;

export function DirPicker(props: DirPickerProps) {
  return <PathPicker mode="directory" {...props} />;
}
