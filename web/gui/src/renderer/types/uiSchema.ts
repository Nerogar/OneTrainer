export type Condition =
  | { field: string; eq: string }
  | { field: string; neq: string }
  | { field: string; in: string[] }
  | { field: string; notIn: string[] }
  | { predicate: string }
  | { and: Condition[] }
  | { or: Condition[] }
  | { not: Condition };

export type FieldType =
  | "entry"
  | "file"
  | "dir"
  | "toggle"
  | "select"
  | "select-kv"
  | "select-adv"
  | "time-entry"
  | "layer-filter"
  | "button"
  | "label";

export interface FieldDef {
  key: string;
  label: string;
  type: FieldType;
  tooltip?: string;
  inputType?: "text" | "number";
  nullable?: boolean;
  enumRef?: string;
  dtypeSubset?: string;
  options?: Array<{ label: string; value: string }>;
  modal?: string;
  visibility?: Condition;
  valuePath?: string;
  unitPath?: string;
  filterPath?: string;
  presetPath?: string;
  regexPath?: string;
  action?: string;
  stringOptions?: string[];
}

export interface SectionDef {
  id: string;
  label: string;
  fields: FieldDef[];
  visibility?: Condition;
}

export interface ColumnDef {
  sections: string[];
}

export interface VariantDef {
  when: Condition;
  columns: ColumnDef[];
}

export interface TabDef {
  id: string;
  label: string;
  renderer: "schema" | "custom";
  layout?: "single-column" | "three-column";
  visibility?: Condition;
  sections?: SectionDef[];
  variants?: VariantDef[];
  sectionDefs?: Record<string, SectionDef>;
}

export interface ModalDef {
  title: string;
  renderer?: string;
  sections?: SectionDef[];
}

export interface EnumDef {
  values: string[];
  labels: Record<string, string>;
}

export interface UiSchema {
  $version: number;
  generated_at?: string;
  predicates: Record<string, string[]>;
  enums: Record<string, EnumDef>;
  dtypeSubsets: Record<string, Array<{ label: string; value: string }>>;
  tabs: TabDef[];
  modals: Record<string, ModalDef>;
  optimizerDefaults: Record<string, Record<string, unknown>>;
  optimizerKeyDetails: Record<string, { title: string; tooltip: string; type: string }>;
  trainingMethodsByModel: Record<string, string[]>;
}
