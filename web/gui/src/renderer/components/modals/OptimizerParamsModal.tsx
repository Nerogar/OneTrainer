import { useEffect, useMemo, useState } from "react";

import { Button, FormEntry, Toggle } from "@/components/shared";
import { useConfigField } from "@/hooks/useConfigField";
import { useConfigStore } from "@/store/configStore";
import type { Optimizer } from "@/types/generated/enums";

import { ModalBase } from "./ModalBase";
import { MuonAdamModal } from "./MuonAdamModal";

export interface OptimizerParamsModalProps {
  open: boolean;
  onClose: () => void;
}

export function OptimizerParamsModal({ open, onClose }: OptimizerParamsModalProps) {
  const [optimizer] = useConfigField<Optimizer>("optimizer.optimizer");
  const [muonAdamOpen, setMuonAdamOpen] = useState(false);
  const changeOptimizer = useConfigStore((s) => s.changeOptimizer);
  const paramsData = useConfigStore((s) => s.optimizerParams);
  const paramsFetched = useConfigStore((s) => s.optimizerParamsFetched);
  const fetchOptimizerParams = useConfigStore((s) => s.fetchOptimizerParams);

  const optimizerName = optimizer as string | undefined;

  // Fetch optimizer metadata once (cached in the store)
  useEffect(() => {
    void fetchOptimizerParams();
  }, [fetchOptimizerParams]);

  const loading = !paramsData && !paramsFetched;

  // Get the parameter keys for the current optimizer
  const optimizerInfo = paramsData && optimizerName ? paramsData.optimizers[optimizerName] : null;
  const detailMap = useMemo(() => paramsData?.detail_map ?? {}, [paramsData]);

  // Split parameters into columns by type, filtering to only those with metadata
  const { numericParams, boolParams, strParams } = useMemo(() => {
    if (!optimizerInfo) return { numericParams: [] as string[], boolParams: [] as string[], strParams: [] as string[] };

    const numeric: string[] = [];
    const bool: string[] = [];
    const str: string[] = [];

    for (const key of optimizerInfo.keys) {
      // Skip muon_adam_config — has its own modal
      if (key === "muon_adam_config") continue;

      const detail = detailMap[key];
      if (!detail) continue;

      if (detail.type === "bool") {
        bool.push(key);
      } else if (detail.type === "str") {
        str.push(key);
      } else {
        numeric.push(key);
      }
    }
    return { numericParams: numeric, boolParams: bool, strParams: str };
  }, [optimizerInfo, detailMap]);

  const isMuonFamily = optimizerName && (optimizerName.includes("MUON") || optimizerName.includes("ADAMUON"));

  return (
    <ModalBase open={open} onClose={onClose} title={`${optimizer ?? "Optimizer"} Parameters`} size="lg">
      <div className="flex gap-2 mb-4">
        <Button
          variant="secondary"
          size="sm"
          onClick={() => {
            if (optimizerName) changeOptimizer(optimizerName);
          }}
        >
          Load Defaults
        </Button>
        {isMuonFamily && (
          <Button variant="secondary" size="sm" onClick={() => setMuonAdamOpen(true)}>
            Muon + Adam Settings
          </Button>
        )}
      </div>

      {loading && (
        <p className="text-sm text-[var(--color-on-surface-secondary)] py-8 text-center">
          Loading optimizer parameters...
        </p>
      )}

      {!loading && !optimizerInfo && (
        <p className="text-sm text-[var(--color-on-surface-secondary)] py-8 text-center">
          No parameter metadata available for this optimizer.
        </p>
      )}

      {!loading && optimizerInfo && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-3">
          {numericParams.map((param) => (
            <FormEntry
              key={param}
              label={detailMap[param]?.title ?? param}
              configPath={`optimizer.${param}`}
              type="number"
              tooltip={detailMap[param]?.tooltip}
              nullable
            />
          ))}
          {strParams.map((param) => (
            <FormEntry
              key={param}
              label={detailMap[param]?.title ?? param}
              configPath={`optimizer.${param}`}
              tooltip={detailMap[param]?.tooltip}
              nullable
            />
          ))}
          {boolParams.map((param) => (
            <Toggle
              key={param}
              configPath={`optimizer.${param}`}
              label={detailMap[param]?.title ?? param}
              tooltip={detailMap[param]?.tooltip}
            />
          ))}
        </div>
      )}

      <div className="flex justify-end mt-6 pt-4 border-t border-[var(--color-border-subtle)]">
        <Button variant="secondary" onClick={onClose}>
          Close
        </Button>
      </div>
      <MuonAdamModal open={muonAdamOpen} onClose={() => setMuonAdamOpen(false)} />
    </ModalBase>
  );
}
