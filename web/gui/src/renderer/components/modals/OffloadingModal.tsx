import { Button, FormEntry, Select, Toggle } from "@/components/shared";
import { GradientCheckpointingMethodValues } from "@/types/generated/enums";

import { ModalBase } from "./ModalBase";

export interface OffloadingModalProps {
  open: boolean;
  onClose: () => void;
}

export function OffloadingModal({ open, onClose }: OffloadingModalProps) {
  return (
    <ModalBase open={open} onClose={onClose} title="Offloading & Checkpointing" size="sm">
      <div className="flex flex-col gap-4">
        <Select
          label="Gradient Checkpointing"
          configPath="gradient_checkpointing"
          options={[...GradientCheckpointingMethodValues]}
          tooltip="Enables gradient checkpointing. This reduces memory usage, but increases training time"
        />
        <FormEntry
          label="Layer Offload Fraction"
          configPath="layer_offload_fraction"
          type="number"
          tooltip="Fraction of layers to offload to CPU (0 = none, 1 = all)"
        />
        <Toggle
          configPath="enable_async_offloading"
          label="Async Offloading"
          tooltip="Enable asynchronous layer offloading for better performance"
        />
        <Toggle
          configPath="enable_activation_offloading"
          label="Activation Offloading"
          tooltip="Offload activations to CPU to save GPU memory"
        />
      </div>

      <div className="flex justify-end mt-6 pt-4 border-t border-[var(--color-border-subtle)]">
        <Button variant="secondary" onClick={onClose}>
          Close
        </Button>
      </div>
    </ModalBase>
  );
}
