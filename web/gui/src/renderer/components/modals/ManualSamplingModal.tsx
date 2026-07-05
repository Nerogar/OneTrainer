import { useEffect, useRef, useState } from "react";

import { trainingApi } from "@/api/trainingApi";
import { Button } from "@/components/shared";
import { useTrainingStore } from "@/store/trainingStore";

import { ModalBase } from "./ModalBase";
import { DEFAULT_SAMPLE, SampleParamsForm, type SampleState } from "./sampleForm";

export interface ManualSamplingModalProps {
  open: boolean;
  onClose: () => void;
}

export function ManualSamplingModal({ open, onClose }: ManualSamplingModalProps) {
  const [sample, setSample] = useState<SampleState>({ ...DEFAULT_SAMPLE });
  const [isSampling, setIsSampling] = useState(false);
  const [sampleImage, setSampleImage] = useState<string | null>(null);
  const waitingRef = useRef(false);
  const latestSample = useTrainingStore((s) => s.latestSample);

  useEffect(() => {
    if (waitingRef.current && latestSample) {
      setSampleImage(latestSample);
      waitingRef.current = false;
    }
  }, [latestSample]);

  const update = <K extends keyof SampleState>(field: K, value: SampleState[K]) => {
    setSample((prev) => ({ ...prev, [field]: value }));
  };

  const handleSample = async () => {
    setIsSampling(true);
    waitingRef.current = true;
    try {
      await trainingApi.sampleCustom({
        prompt: sample.prompt,
        negative_prompt: sample.negative_prompt,
        height: sample.height,
        width: sample.width,
        seed: sample.seed,
        random_seed: sample.random_seed,
        diffusion_steps: sample.diffusion_steps,
        cfg_scale: sample.cfg_scale,
      });
    } catch {
      waitingRef.current = false;
    } finally {
      setIsSampling(false);
    }
  };

  return (
    <ModalBase open={open} onClose={onClose} title="Manual Sample" size="xl" closeOnBackdrop={false}>
      <SampleParamsForm
        state={sample}
        onChange={update}
        sampleImage={sampleImage}
        isSampling={isSampling}
        placeholder="No sample generated yet"
      />

      <div className="flex justify-between mt-6 pt-4 border-t border-[var(--color-border-subtle)]">
        <Button variant="primary" onClick={handleSample} disabled={isSampling}>
          {isSampling ? "Sampling..." : "Sample"}
        </Button>
        <Button variant="secondary" onClick={onClose}>
          Close
        </Button>
      </div>
    </ModalBase>
  );
}
