import { useCallback, useMemo, useState } from "react";
import { lazy, Suspense } from "react";

import { SchemaField, SectionCard } from "@/components/shared";
import { useConfigStore } from "@/store/configStore";
import { useUiSchemaStore } from "@/store/uiSchemaStore";
import type { LossWeight, ModelType } from "@/types/generated/enums";
import { LOSS_WEIGHT_SUPPORTS_FLOW_MATCHING } from "@/types/generated/lossWeightInfo";
import { MODEL_TYPE_GROUPS } from "@/types/generated/modelTypeInfo";
import type { FieldDef, SectionDef, TabDef } from "@/types/uiSchema";
import { evaluateCondition } from "@/utils/conditionEval";

import { SchemaModal } from "./SchemaModal";

const OptimizerParamsModal = lazy(() =>
  import("@/components/modals/OptimizerParamsModal").then((m) => ({ default: m.OptimizerParamsModal })),
);
const SchedulerParamsModal = lazy(() =>
  import("@/components/modals/SchedulerParamsModal").then((m) => ({ default: m.SchedulerParamsModal })),
);
const TimestepDistModal = lazy(() =>
  import("@/components/modals/TimestepDistModal").then((m) => ({ default: m.TimestepDistModal })),
);
const OffloadingModal = lazy(() =>
  import("@/components/modals/OffloadingModal").then((m) => ({ default: m.OffloadingModal })),
);

const CUSTOM_MODAL_RENDERERS: Record<
  string,
  React.LazyExoticComponent<React.ComponentType<{ open: boolean; onClose: () => void }>>
> = {
  "optimizer-params": OptimizerParamsModal,
  "scheduler-params": SchedulerParamsModal,
  "timestep-dist": TimestepDistModal,
  offloading: OffloadingModal,
};

interface SchemaTabRendererProps {
  tab: TabDef;
}

export function SchemaTabRenderer({ tab }: SchemaTabRendererProps) {
  const schema = useUiSchemaStore((s) => s.schema);
  const config = useConfigStore((s) => s.config);
  const [activeModal, setActiveModal] = useState<string | null>(null);

  const predicates = useMemo(() => schema?.predicates ?? {}, [schema]);
  const configRecord = useMemo(() => (config ?? {}) as Record<string, unknown>, [config]);

  const resolvedSections = useMemo(() => {
    const defs = tab.sectionDefs ?? {};

    if (tab.variants && tab.variants.length > 0 && tab.sectionDefs) {
      let matchedVariant = tab.variants[0];
      for (const variant of tab.variants) {
        if (evaluateCondition(variant.when, configRecord, predicates)) {
          matchedVariant = variant;
          break;
        }
      }

      if (tab.layout === "three-column") {
        return matchedVariant.columns.map((col) => col.sections.map((sectionId) => defs[sectionId]).filter(Boolean));
      }
      const allSections = matchedVariant.columns.flatMap((col) =>
        col.sections.map((sectionId) => defs[sectionId]).filter(Boolean),
      );
      return [allSections];
    }

    return tab.sections ? [tab.sections] : [];
  }, [tab, configRecord, predicates]);

  const resolveEnumOptions = useCallback(
    (field: FieldDef): string[] => {
      if (field.enumRef && schema?.enums[field.enumRef]) {
        const values = schema.enums[field.enumRef].values;
        if (field.key === "loss_weight_fn") {
          const modelType = configRecord["model_type"] as ModelType | undefined;
          if (modelType) {
            const flowMatching = MODEL_TYPE_GROUPS.is_flow_matching ?? [];
            const isFlowMatching = flowMatching.includes(modelType);
            return values.filter(
              (v) => v === "CONSTANT" || LOSS_WEIGHT_SUPPORTS_FLOW_MATCHING[v as LossWeight] === isFlowMatching,
            );
          }
        }
        return values;
      }
      return [];
    },
    [schema, configRecord],
  );

  const resolveKvOptions = useCallback(
    (field: FieldDef): Array<{ label: string; value: string }> | undefined => {
      if (field.dtypeSubset && schema?.dtypeSubsets[field.dtypeSubset]) {
        return schema.dtypeSubsets[field.dtypeSubset];
      }
      return field.options;
    },
    [schema],
  );

  const handleAdvancedClick = useCallback(
    (fieldKey: string) => {
      // Walk resolved sections to find the field and its modal ref
      for (const group of resolvedSections) {
        for (const section of group) {
          for (const field of section.fields) {
            if (field.key === fieldKey && field.modal) {
              setActiveModal(field.modal);
              return;
            }
          }
        }
      }
    },
    [resolvedSections],
  );

  const closeModal = useCallback(() => setActiveModal(null), []);

  const visibleFields = useCallback(
    (fields: FieldDef[]) => fields.filter((f) => evaluateCondition(f.visibility, configRecord, predicates)),
    [configRecord, predicates],
  );

  const isSectionVisible = useCallback(
    (section: SectionDef) => evaluateCondition(section.visibility, configRecord, predicates),
    [configRecord, predicates],
  );

  const renderSection = (section: SectionDef) => {
    if (!isSectionVisible(section)) return null;
    const fields = visibleFields(section.fields);
    if (fields.length === 0) return null;

    return (
      <SectionCard key={section.id} title={section.label}>
        <div
          className={tab.layout === "three-column" ? "flex flex-col gap-4" : "grid grid-cols-1 md:grid-cols-2 gap-4"}
        >
          {fields.map((field) => (
            <SchemaField
              key={field.key}
              field={field}
              onAdvancedClick={handleAdvancedClick}
              resolveOptions={resolveEnumOptions}
              resolveKvOptions={resolveKvOptions}
            />
          ))}
        </div>
      </SectionCard>
    );
  };

  const activeModalDef = activeModal ? schema?.modals[activeModal] : undefined;
  const CustomModalComponent = activeModal ? CUSTOM_MODAL_RENDERERS[activeModalDef?.renderer ?? ""] : undefined;

  return (
    <>
      {tab.layout === "three-column" ? (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {resolvedSections.map((column, colIdx) => (
            <div key={colIdx} className="flex flex-col gap-6">
              {column.map((section) => renderSection(section))}
            </div>
          ))}
        </div>
      ) : (
        <div className="flex flex-col gap-6">{resolvedSections.flat().map((section) => renderSection(section))}</div>
      )}

      {CustomModalComponent && (
        <Suspense fallback={null}>
          <CustomModalComponent open={!!activeModal} onClose={closeModal} />
        </Suspense>
      )}

      {activeModalDef && !activeModalDef.renderer && (
        <SchemaModal
          open={!!activeModal}
          onClose={closeModal}
          modal={activeModalDef}
          configRecord={configRecord}
          predicates={predicates}
        />
      )}
    </>
  );
}
