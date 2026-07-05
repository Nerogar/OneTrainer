import { useCallback, useEffect, useState } from "react";

import { Button } from "@/components/shared";
import { INPUT_FULL } from "@/utils/inputStyles";

import { ModalBase } from "./ModalBase";

export interface SavePresetModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (name: string) => void;
}

export function SavePresetModal({ isOpen, onClose, onSave }: SavePresetModalProps) {
  const [name, setName] = useState("");

  useEffect(() => {
    if (!isOpen) setName("");
  }, [isOpen]);

  const handleSave = useCallback(() => {
    const trimmed = name.trim();
    if (trimmed) {
      onSave(trimmed);
      setName("");
    }
  }, [name, onSave]);

  const handleClose = useCallback(() => {
    setName("");
    onClose();
  }, [onClose]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && name.trim()) {
      e.preventDefault();
      handleSave();
    }
  };

  return (
    <ModalBase open={isOpen} onClose={handleClose} title="Save Preset" size="sm">
      <div className="flex flex-col gap-4">
        <label className="flex flex-col gap-1.5">
          <span className="text-sm font-medium text-[var(--color-on-surface)]">Preset Name</span>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={handleKeyDown}
            className={INPUT_FULL}
            placeholder="My preset"
            autoFocus
          />
        </label>
      </div>

      <div className="flex items-center justify-end gap-3 mt-6 pt-4 border-t border-[var(--color-border-subtle)]">
        <Button variant="secondary" size="md" onClick={handleClose}>
          Cancel
        </Button>
        <Button variant="primary" size="md" onClick={handleSave} disabled={!name.trim()}>
          Save
        </Button>
      </div>
    </ModalBase>
  );
}
