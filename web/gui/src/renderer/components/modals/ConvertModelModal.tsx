import { useState } from "react";

import { toolsApi } from "@/api/toolsApi";
import { Button, FilePicker, Select } from "@/components/shared";
import {
  CONVERT_MODEL_TYPES,
  CONVERT_OUTPUT_DTYPES,
  CONVERT_OUTPUT_FORMATS,
  CONVERT_TRAINING_METHODS,
} from "@/types/generated/dropdownSources";

import { ModalBase } from "./ModalBase";

export interface ConvertModelModalProps {
  open: boolean;
  onClose: () => void;
}

type StatusKind = "ready" | "converting" | "success" | "error";

interface Status {
  kind: StatusKind;
  message: string;
}

const STATUS_READY: Status = { kind: "ready", message: "Ready" };
const STATUS_CONVERTING: Status = { kind: "converting", message: "Converting..." };

export function ConvertModelModal({ open, onClose }: ConvertModelModalProps) {
  const [modelType, setModelType] = useState("STABLE_DIFFUSION_15");
  const [trainingMethod, setTrainingMethod] = useState("FINE_TUNE");
  const [inputName, setInputName] = useState("");
  const [outputDtype, setOutputDtype] = useState("FLOAT_16");
  const [outputFormat, setOutputFormat] = useState("SAFETENSORS");
  const [outputDestination, setOutputDestination] = useState("");
  const [status, setStatus] = useState<Status>(STATUS_READY);

  const isConverting = status.kind === "converting";

  const handleConvert = async () => {
    if (!inputName.trim()) {
      setStatus({ kind: "error", message: "Input name is required." });
      return;
    }
    if (!outputDestination.trim()) {
      setStatus({ kind: "error", message: "Output destination is required." });
      return;
    }

    setStatus(STATUS_CONVERTING);
    try {
      const result = await toolsApi.convertModel({
        model_type: modelType,
        training_method: trainingMethod,
        input_name: inputName,
        output_dtype: outputDtype,
        output_model_format: outputFormat,
        output_model_destination: outputDestination,
      });
      if (result.ok) {
        setStatus({ kind: "success", message: "Model converted successfully." });
      } else {
        setStatus({ kind: "error", message: result.error ?? "Unknown error during conversion." });
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setStatus({ kind: "error", message });
    }
  };

  const statusColor = (() => {
    switch (status.kind) {
      case "ready":
        return "var(--color-on-surface-secondary)";
      case "converting":
        return "var(--color-cobalt-600)";
      case "success":
        return "var(--color-success-500)";
      case "error":
        return "var(--color-error-500)";
    }
  })();

  return (
    <ModalBase open={open} onClose={onClose} title="Convert Model" size="md" closeOnBackdrop={!isConverting}>
      <div className="flex flex-col gap-4">
        <Select
          label="Model Type"
          options={CONVERT_MODEL_TYPES}
          value={modelType}
          onChange={setModelType}
          disabled={isConverting}
          tooltip="Type of the model"
        />

        <Select
          label="Training Method"
          options={CONVERT_TRAINING_METHODS}
          value={trainingMethod}
          onChange={setTrainingMethod}
          disabled={isConverting}
          tooltip="The type of model to convert"
        />

        <FilePicker
          label="Input Name"
          value={inputName}
          onChange={setInputName}
          disabled={isConverting}
          tooltip="Filename, directory or Hugging Face repository of the model"
        />

        <Select
          label="Output Data Type"
          options={CONVERT_OUTPUT_DTYPES}
          value={outputDtype}
          onChange={setOutputDtype}
          disabled={isConverting}
          tooltip="Precision to use when saving the output model"
        />

        <Select
          label="Output Format"
          options={CONVERT_OUTPUT_FORMATS}
          value={outputFormat}
          onChange={setOutputFormat}
          disabled={isConverting}
          tooltip="Format to use when saving the output model"
        />

        <FilePicker
          label="Model Output Destination"
          value={outputDestination}
          onChange={setOutputDestination}
          disabled={isConverting}
          tooltip="Filename or directory where the output model is saved"
        />
      </div>

      <div className="flex items-center justify-between mt-6 pt-4 border-t border-[var(--color-border-subtle)]">
        <span
          className="text-sm font-medium truncate max-w-[60%]"
          style={{ color: statusColor }}
          title={status.message}
        >
          {status.message}
        </span>

        <div className="flex gap-3">
          <Button variant="primary" size="md" onClick={handleConvert} disabled={isConverting} loading={isConverting}>
            {isConverting ? "Converting..." : "Convert"}
          </Button>
          <Button variant="secondary" size="md" onClick={onClose} disabled={isConverting}>
            Close
          </Button>
        </div>
      </div>
    </ModalBase>
  );
}
