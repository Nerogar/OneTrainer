import { ModalBase } from "@/components/modals/ModalBase";
import { SchemaField } from "@/components/shared";
import { SectionCard } from "@/components/shared";
import type { ModalDef } from "@/types/uiSchema";
import { evaluateCondition } from "@/utils/conditionEval";

interface SchemaModalProps {
  open: boolean;
  onClose: () => void;
  modal: ModalDef;
  configRecord: Record<string, unknown>;
  predicates: Record<string, string[]>;
}

/**
 * Generic modal renderer: renders modal sections/fields from a schema ModalDef.
 * Used for simple modals that are just collections of fields (Timestep, Offloading, etc.).
 * Complex modals (Optimizer, Scheduler) use custom renderers instead.
 */
export function SchemaModal({ open, onClose, modal, configRecord, predicates }: SchemaModalProps) {
  if (!modal.sections || modal.sections.length === 0) return null;

  return (
    <ModalBase open={open} onClose={onClose} title={modal.title}>
      <div className="flex flex-col gap-6">
        {modal.sections.map((section) => {
          const visibleFields = section.fields.filter((f) => evaluateCondition(f.visibility, configRecord, predicates));
          if (visibleFields.length === 0) return null;

          // If section has a label, wrap in SectionCard; otherwise render flat
          if (section.label) {
            return (
              <SectionCard key={section.id} title={section.label}>
                <div className="flex flex-col gap-4">
                  {visibleFields.map((field) => (
                    <SchemaField key={field.key} field={field} />
                  ))}
                </div>
              </SectionCard>
            );
          }

          return (
            <div key={section.id} className="flex flex-col gap-4">
              {visibleFields.map((field) => (
                <SchemaField key={field.key} field={field} />
              ))}
            </div>
          );
        })}
      </div>
    </ModalBase>
  );
}
