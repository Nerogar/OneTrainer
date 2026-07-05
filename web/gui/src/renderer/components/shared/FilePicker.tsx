import { PathPicker, type PathPickerProps } from "./PathPicker";

export type FilePickerProps = Omit<PathPickerProps, "mode">;

export function FilePicker(props: FilePickerProps) {
  return <PathPicker mode="file" {...props} />;
}
