import { FormEntry, Toggle } from "@/components/shared";

import { ModalBase } from "./ModalBase";

export interface MuonAdamModalProps {
  open: boolean;
  onClose: () => void;
}

export function MuonAdamModal({ open, onClose }: MuonAdamModalProps) {
  return (
    <ModalBase open={open} onClose={onClose} title="Muon + Adam Settings" size="md">
      <div className="flex flex-col gap-4">
        <h4 className="text-sm font-semibold text-[var(--color-on-surface-secondary)] uppercase tracking-wide">
          Muon Parameters
        </h4>
        <FormEntry
          label="Muon Adam LR"
          configPath="optimizer.muon_adam_lr"
          type="number"
          tooltip="Learning rate for Adam portion of Muon optimizer"
          nullable
        />
        <FormEntry
          label="Muon TE1 Adam LR"
          configPath="optimizer.muon_te1_adam_lr"
          type="number"
          tooltip="Adam LR for Text Encoder 1"
          nullable
        />
        <FormEntry
          label="Muon TE2 Adam LR"
          configPath="optimizer.muon_te2_adam_lr"
          type="number"
          tooltip="Adam LR for Text Encoder 2"
          nullable
        />
        <Toggle
          configPath="optimizer.MuonWithAuxAdam"
          label="Muon with Auxiliary Adam"
          tooltip="Use auxiliary Adam optimizer alongside Muon"
        />
        <FormEntry
          label="NS Steps"
          configPath="optimizer.ns_steps"
          type="number"
          tooltip="Newton-Schulz iteration steps"
          nullable
        />
        <Toggle
          configPath="optimizer.accelerated_ns"
          label="Accelerated NS"
          tooltip="Use accelerated Newton-Schulz iterations"
        />

        <h4 className="text-sm font-semibold text-[var(--color-on-surface-secondary)] uppercase tracking-wide mt-4">
          Adam Regex Filter
        </h4>
        <Toggle
          configPath="optimizer.muon_adam_regex"
          label="Use Adam Regex"
          tooltip="Use regex to select which parameters use Adam vs Muon"
        />

        <h4 className="text-sm font-semibold text-[var(--color-on-surface-secondary)] uppercase tracking-wide mt-4">
          Normuon Variant
        </h4>
        <Toggle configPath="optimizer.normuon_variant" label="Normuon Variant" tooltip="Use the Normuon variant" />
        <FormEntry
          label="Beta2 Normuon"
          configPath="optimizer.beta2_normuon"
          type="number"
          tooltip="Beta2 for Normuon variant"
          nullable
        />
        <FormEntry
          label="Normuon Epsilon"
          configPath="optimizer.normuon_eps"
          type="number"
          tooltip="Epsilon for Normuon variant"
          nullable
        />

        <h4 className="text-sm font-semibold text-[var(--color-on-surface-secondary)] uppercase tracking-wide mt-4">
          Orthogonalization
        </h4>
        <Toggle configPath="optimizer.low_rank_ortho" label="Low Rank Ortho" tooltip="Use low-rank orthogonalization" />
        <FormEntry
          label="Ortho Rank"
          configPath="optimizer.ortho_rank"
          type="number"
          tooltip="Rank for low-rank orthogonalization"
          nullable
        />
      </div>

      <div className="flex justify-end mt-6 pt-4 border-t border-[var(--color-border-subtle)]">
        <button
          onClick={onClose}
          className="px-4 py-2 rounded-[var(--radius-sm)] text-sm font-medium bg-transparent border border-[var(--color-border-subtle)] text-[var(--color-on-surface)] hover:border-[var(--color-cobalt-600)] transition-colors duration-200 cursor-pointer"
        >
          Close
        </button>
      </div>
    </ModalBase>
  );
}
