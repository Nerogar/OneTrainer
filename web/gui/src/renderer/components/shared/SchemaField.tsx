import type { FieldDef } from "@/types/uiSchema";

import { DirPicker } from "./DirPicker";
import { FilePicker } from "./FilePicker";
import { FormEntry } from "./FormEntry";
import { LayerFilterEntry } from "./LayerFilterEntry";
import { Select } from "./Select";
import { TimeEntry } from "./TimeEntry";
import { Toggle } from "./Toggle";

export interface SchemaFieldProps {
  field: FieldDef;
  /** Called when a "select-adv" field's advanced button is clicked. */
  onAdvancedClick?: (fieldKey: string) => void;
  /** Resolve dynamic string options at render time (e.g. from schema.enums). */
  resolveOptions?: (field: FieldDef) => string[];
  /** Resolve key-value options at render time (e.g. from schema.dtypeSubsets). */
  resolveKvOptions?: (field: FieldDef) => Array<{ label: string; value: string }> | undefined;
}

export function SchemaField({ field, onAdvancedClick, resolveOptions, resolveKvOptions }: SchemaFieldProps) {
  const stringOptions = resolveOptions ? resolveOptions(field) : (field.stringOptions ?? (field.enumRef ? [] : []));
  const kvOptions = resolveKvOptions ? resolveKvOptions(field) : field.options;

  switch (field.type) {
    case "entry":
      return (
        <FormEntry
          label={field.label}
          configPath={field.key}
          type={field.inputType ?? "text"}
          tooltip={field.tooltip}
          nullable={field.nullable}
        />
      );
    case "file":
      return <FilePicker label={field.label} configPath={field.key} tooltip={field.tooltip} />;
    case "dir":
      return <DirPicker label={field.label} configPath={field.key} tooltip={field.tooltip} />;
    case "toggle":
      return <Toggle configPath={field.key} label={field.label} tooltip={field.tooltip} />;
    case "select":
      return <Select label={field.label} configPath={field.key} options={stringOptions} tooltip={field.tooltip} />;
    case "select-kv":
      return <Select label={field.label} configPath={field.key} options={kvOptions ?? []} tooltip={field.tooltip} />;
    case "select-adv":
      return (
        <Select
          label={field.label}
          configPath={field.key}
          options={stringOptions}
          tooltip={field.tooltip}
          onAdvancedClick={() => onAdvancedClick?.(field.key)}
        />
      );
    case "time-entry":
      return (
        <TimeEntry
          label={field.label}
          valuePath={field.valuePath ?? field.key}
          unitPath={field.unitPath ?? `${field.key}_unit`}
          tooltip={field.tooltip}
        />
      );
    case "layer-filter":
      return (
        <LayerFilterEntry
          filterPath={field.filterPath ?? "layer_filter"}
          presetPath={field.presetPath ?? "layer_filter_preset"}
          regexPath={field.regexPath ?? "layer_filter_regex"}
        />
      );
    default:
      return null;
  }
}
