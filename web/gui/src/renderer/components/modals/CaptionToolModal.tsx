import { Eye, Loader2, Plus, Trash2 } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import { type ApiCaptionRequest, type PreviewResponse, toolsApi } from "@/api/toolsApi";
import { Button, DirPicker, FormEntry, ProgressBar, Select, SliderEntry, Toggle, Tooltip } from "@/components/shared";
import { useToolPolling } from "@/hooks/useToolPolling";
import { CAPTION_MODELS, CAPTION_MODES } from "@/types/generated/dropdownSources";
import { INPUT_FLEX, TEXTAREA_FULL } from "@/utils/inputStyles";

import { ModalBase } from "./ModalBase";

export interface CaptionToolModalProps {
  open: boolean;
  onClose: () => void;
}

const LOCAL_MODELS = new Set(["Blip", "Blip2", "WD14 VIT v2"]);
const API_MODELS = new Set(["OpenAI Compatible", "Gemini API"]);

interface CaptionState {
  model: string;
  folder: string;
  initial_caption: string;
  caption_prefix: string;
  caption_postfix: string;
  mode: string;
  include_subdirectories: boolean;
  // API fields
  api_url: string;
  api_key: string;
  model_name: string;
  system_prompt: string;
  temperature: number;
  max_tokens: number;
  enable_thinking: boolean;
  batch_size: number;
  requests_per_minute: number;
  // Prompt
  prompt: string;
  additional_prompts: string[];
  pass_filename: boolean;
  pass_metadata: boolean;
  pass_current_caption: boolean;
}

const DEFAULT_STATE: CaptionState = {
  model: "Blip",
  folder: "",
  initial_caption: "",
  caption_prefix: "",
  caption_postfix: "",
  mode: "fill",
  include_subdirectories: false,
  api_url: "http://localhost:1234/v1/chat/completions",
  api_key: "",
  model_name: "",
  system_prompt: "",
  temperature: 0.6,
  max_tokens: -1,
  enable_thinking: false,
  batch_size: 1,
  requests_per_minute: 0,
  prompt: "",
  additional_prompts: [],
  pass_filename: false,
  pass_metadata: false,
  pass_current_caption: false,
};

function isLocalModel(model: string): boolean {
  return LOCAL_MODELS.has(model);
}

function isApiModel(model: string): boolean {
  return API_MODELS.has(model);
}

export function CaptionToolModal({ open, onClose }: CaptionToolModalProps) {
  const [state, setState] = useState<CaptionState>({ ...DEFAULT_STATE });
  const polling = useToolPolling();
  const { status, isRunning, error } = polling;
  const [preview, setPreview] = useState<PreviewResponse | null>(null);
  const [isPreviewing, setIsPreviewing] = useState(false);
  const [showApiKey, setShowApiKey] = useState(false);

  const update = useCallback(<K extends keyof CaptionState>(field: K, value: CaptionState[K]) => {
    setState((prev) => ({ ...prev, [field]: value }));
  }, []);

  useEffect(() => {
    if (!open || !isApiModel(state.model)) return;
    toolsApi
      .getCaptionKeys()
      .then((keys) => {
        setState((prev) => ({
          ...prev,
          api_key: prev.api_key || keys.openai_api_key || keys.gemini_api_key || "",
          api_url: prev.api_url || keys.openai_api_url || "http://localhost:1234/v1/chat/completions",
        }));
      })
      .catch(() => {
        /* ignore */
      });
  }, [open, state.model]);

  const handleGenerate = async () => {
    if (!state.folder) {
      polling.setError("Please select a folder.");
      return;
    }
    polling.start();

    try {
      let result;
      if (isApiModel(state.model)) {
        const params: ApiCaptionRequest = {
          backend: state.model === "OpenAI Compatible" ? "openai" : "gemini",
          folder: state.folder,
          prompt: state.prompt || "Describe this image.",
          additional_prompts: state.additional_prompts.filter((p) => p.trim()),
          mode: state.mode,
          include_subdirectories: state.include_subdirectories,
          caption_prefix: state.caption_prefix,
          caption_postfix: state.caption_postfix,
          api_url: state.api_url,
          api_key: state.api_key,
          model_name: state.model_name,
          system_prompt: state.system_prompt,
          temperature: state.temperature,
          max_tokens: state.max_tokens,
          enable_thinking: state.enable_thinking,
          batch_size: state.batch_size,
          requests_per_minute: state.requests_per_minute,
          pass_filename: state.pass_filename,
          pass_metadata: state.pass_metadata,
          pass_current_caption: state.pass_current_caption,
        };
        result = await toolsApi.generateCaptionsApi(params);
      } else {
        result = await toolsApi.generateCaptions({
          model: state.model,
          folder: state.folder,
          initial_caption: state.initial_caption,
          caption_prefix: state.caption_prefix,
          caption_postfix: state.caption_postfix,
          mode: state.mode,
          include_subdirectories: state.include_subdirectories,
        });
      }
      if (!result.ok) {
        polling.setError(result.error ?? "Failed to start");
        polling.stop();
      }
    } catch (err) {
      polling.setError(err instanceof Error ? err.message : "Unknown error");
      polling.stop();
    }
  };

  const handlePreview = async () => {
    if (!state.folder) {
      polling.setError("Please select a folder.");
      return;
    }
    polling.setError(null);
    setIsPreviewing(true);
    setPreview(null);
    try {
      const result = await toolsApi.previewCaption({
        backend: isApiModel(state.model) ? (state.model === "OpenAI Compatible" ? "openai" : "gemini") : "",
        model: isLocalModel(state.model) ? state.model : "",
        folder: state.folder,
        prompt: state.prompt || "Describe this image.",
        include_subdirectories: state.include_subdirectories,
        api_url: state.api_url,
        api_key: state.api_key,
        model_name: state.model_name,
        system_prompt: state.system_prompt,
        temperature: state.temperature,
        max_tokens: state.max_tokens,
        enable_thinking: state.enable_thinking,
        pass_filename: state.pass_filename,
        pass_metadata: state.pass_metadata,
        pass_current_caption: state.pass_current_caption,
      });
      setPreview(result);
      if (!result.ok && result.error) polling.setError(result.error);
    } catch (err) {
      polling.setError(err instanceof Error ? err.message : "Preview failed");
    } finally {
      setIsPreviewing(false);
    }
  };

  const handleCancel = async () => {
    try {
      await toolsApi.cancel();
      polling.stop();
    } catch {
      /* ignore */
    }
  };

  const addPrompt = () => {
    setState((prev) => ({ ...prev, additional_prompts: [...prev.additional_prompts, ""] }));
  };

  const removePrompt = (index: number) => {
    setState((prev) => ({
      ...prev,
      additional_prompts: prev.additional_prompts.filter((_, i) => i !== index),
    }));
  };

  const updatePrompt = (index: number, value: string) => {
    setState((prev) => ({
      ...prev,
      additional_prompts: prev.additional_prompts.map((p, i) => (i === index ? value : p)),
    }));
  };

  return (
    <ModalBase open={open} onClose={onClose} title="Batch Generate Captions" size="lg" closeOnBackdrop={!isRunning}>
      <div className="flex flex-col gap-4 max-h-[70vh] overflow-y-auto pr-1">
        <Select
          label="Model"
          options={CAPTION_MODELS}
          value={state.model}
          onChange={(v) => update("model", v)}
          disabled={isRunning}
        />

        <DirPicker label="Folder" value={state.folder} onChange={(v) => update("folder", v)} disabled={isRunning} />

        {isLocalModel(state.model) && (
          <>
            <FormEntry
              label="Initial Caption"
              value={state.initial_caption}
              onChange={(v) => update("initial_caption", String(v))}
              placeholder="Optional initial caption..."
              disabled={isRunning}
            />
            <FormEntry
              label="Caption Prefix"
              value={state.caption_prefix}
              onChange={(v) => update("caption_prefix", String(v))}
              placeholder="Optional prefix..."
              disabled={isRunning}
            />
            <FormEntry
              label="Caption Postfix"
              value={state.caption_postfix}
              onChange={(v) => update("caption_postfix", String(v))}
              placeholder="Optional postfix..."
              disabled={isRunning}
            />
          </>
        )}

        {isApiModel(state.model) && (
          <div className="flex flex-col gap-3 p-3 rounded-[var(--radius-sm)] bg-[var(--color-surface-elevated)] border border-[var(--color-border-subtle)]">
            <h4 className="text-sm font-semibold text-[var(--color-on-surface-secondary)]">
              {state.model === "OpenAI Compatible" ? "OpenAI" : "Gemini"} API Settings
            </h4>
            {state.model === "OpenAI Compatible" && (
              <FormEntry
                label="API URL"
                value={state.api_url}
                onChange={(v) => update("api_url", String(v))}
                placeholder="http://localhost:1234/v1/chat/completions"
                disabled={isRunning}
              />
            )}
            <div className="flex flex-col gap-1">
              <label className="text-sm font-medium text-[var(--color-on-surface)]">API Key</label>
              <div className="flex gap-1">
                <input
                  type={showApiKey ? "text" : "password"}
                  value={state.api_key}
                  onChange={(e) => update("api_key", e.target.value)}
                  disabled={isRunning}
                  placeholder={state.model === "OpenAI Compatible" ? "sk-..." : "AIzaSy..."}
                  className="flex-1 px-3 py-1.5 rounded-[var(--radius-sm)] border border-[var(--color-border-subtle)] bg-[var(--color-surface-raised)] text-sm text-[var(--color-on-surface)]"
                />
                <Button variant="secondary" onClick={() => setShowApiKey(!showApiKey)} disabled={isRunning}>
                  {showApiKey ? "Hide" : "Show"}
                </Button>
              </div>
            </div>
            <FormEntry
              label="Model Name"
              value={state.model_name}
              onChange={(v) => update("model_name", String(v))}
              placeholder={state.model === "OpenAI Compatible" ? "local-model" : "gemini-1.5-flash"}
              disabled={isRunning}
            />
            <FormEntry
              label="System Prompt"
              value={state.system_prompt}
              onChange={(v) => update("system_prompt", String(v))}
              placeholder="Optional system prompt..."
              disabled={isRunning}
            />
            <div className="grid grid-cols-2 gap-3">
              <SliderEntry
                label="Temperature"
                value={state.temperature}
                onChange={(v) => update("temperature", v)}
                min={0}
                max={2}
                step={0.1}
                disabled={isRunning}
              />
              <FormEntry
                label="Max Tokens"
                value={state.max_tokens}
                onChange={(v) => update("max_tokens", Number(v))}
                type="number"
                placeholder="-1 = unlimited"
                disabled={isRunning}
              />
              {state.model === "OpenAI Compatible" && (
                <>
                  <FormEntry
                    label="Batch Size"
                    value={state.batch_size}
                    onChange={(v) => update("batch_size", Number(v))}
                    type="number"
                    tooltip="Number of concurrent API requests"
                    disabled={isRunning}
                  />
                  <FormEntry
                    label="RPM Limit"
                    value={state.requests_per_minute}
                    onChange={(v) => update("requests_per_minute", Number(v))}
                    type="number"
                    tooltip="Requests per minute (0 = unlimited)"
                    disabled={isRunning}
                  />
                </>
              )}
            </div>
            {state.model === "OpenAI Compatible" && (
              <Toggle
                label="Enable thinking mode"
                value={state.enable_thinking}
                onChange={(v) => update("enable_thinking", v)}
                disabled={isRunning}
              />
            )}
          </div>
        )}

        {isApiModel(state.model) && (
          <div className="flex flex-col gap-3 p-3 rounded-[var(--radius-sm)] bg-[var(--color-surface-elevated)] border border-[var(--color-border-subtle)]">
            <h4 className="text-sm font-semibold text-[var(--color-on-surface-secondary)]">Prompt</h4>
            <textarea
              value={state.prompt}
              onChange={(e) => update("prompt", e.target.value)}
              disabled={isRunning}
              placeholder="Describe this image."
              rows={3}
              className={`${TEXTAREA_FULL} resize-y`}
            />

            {state.additional_prompts.map((p, i) => (
              <div key={i} className="flex gap-1">
                <input
                  type="text"
                  value={p}
                  onChange={(e) => updatePrompt(i, e.target.value)}
                  disabled={isRunning}
                  placeholder={`Additional prompt ${i + 1}...`}
                  className={INPUT_FLEX}
                />
                <button
                  type="button"
                  onClick={() => removePrompt(i)}
                  disabled={isRunning}
                  className="p-1.5 rounded-[var(--radius-sm)] text-[var(--color-error-500)] hover:bg-[var(--color-error-500)]/10"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            ))}

            <Tooltip content="A prompt will be randomly selected from the pool for each image">
              <Button variant="secondary" onClick={addPrompt} disabled={isRunning}>
                <Plus className="w-4 h-4 mr-1" /> Add Additional Prompt
              </Button>
            </Tooltip>

            <div className="flex flex-col gap-2 pt-2 border-t border-[var(--color-border-subtle)]">
              <h4 className="text-xs font-semibold text-[var(--color-on-surface-secondary)] uppercase tracking-wider">
                Context Hints
              </h4>
              <Toggle
                label="Pass file metadata (XMP/EXIF)"
                value={state.pass_metadata}
                onChange={(v) => update("pass_metadata", v)}
                disabled={isRunning}
              />
              <Toggle
                label="Pass file name"
                value={state.pass_filename}
                onChange={(v) => update("pass_filename", v)}
                disabled={isRunning}
              />
              <Toggle
                label="Pass current caption"
                value={state.pass_current_caption}
                onChange={(v) => update("pass_current_caption", v)}
                disabled={isRunning}
              />
            </div>

            <FormEntry
              label="Caption Prefix"
              value={state.caption_prefix}
              onChange={(v) => update("caption_prefix", String(v))}
              placeholder="Optional prefix..."
              disabled={isRunning}
            />
            <FormEntry
              label="Caption Postfix"
              value={state.caption_postfix}
              onChange={(v) => update("caption_postfix", String(v))}
              placeholder="Optional postfix..."
              disabled={isRunning}
            />
          </div>
        )}

        <Select
          label="Mode"
          options={CAPTION_MODES}
          value={state.mode}
          onChange={(v) => update("mode", v)}
          disabled={isRunning}
        />

        <Toggle
          label="Include subfolders"
          value={state.include_subdirectories}
          onChange={(v) => update("include_subdirectories", v)}
          disabled={isRunning}
        />

        {preview?.ok && (
          <div className="flex flex-col gap-2 p-3 rounded-[var(--radius-sm)] bg-[var(--color-surface-elevated)] border border-[var(--color-cobalt-600)]/30">
            <h4 className="text-sm font-semibold text-[var(--color-cobalt-600)]">Preview</h4>
            <div className="flex gap-3">
              {preview.image_base64 && (
                <img
                  src={`data:image/jpeg;base64,${preview.image_base64}`}
                  alt="Preview"
                  className="w-24 h-24 object-cover rounded-[var(--radius-sm)] flex-shrink-0"
                />
              )}
              <div className="flex-1 min-w-0">
                <p className="text-xs text-[var(--color-on-surface-secondary)] mb-1 truncate">
                  {preview.image_path.split(/[/\\]/).pop()}
                </p>
                <p className="text-sm text-[var(--color-on-surface)] whitespace-pre-wrap break-words">
                  {preview.caption}
                </p>
              </div>
            </div>
            {preview.prompt_used && (
              <details className="text-xs text-[var(--color-on-surface-secondary)]">
                <summary className="cursor-pointer">Prompt used</summary>
                <p className="mt-1 whitespace-pre-wrap">{preview.prompt_used}</p>
              </details>
            )}
          </div>
        )}

        <div className="pt-2">
          <ProgressBar
            value={polling.progress}
            label={polling.progressLabel}
            indeterminate={isRunning && polling.progress === 0}
          />
        </div>

        {error && (
          <div className="p-3 rounded-[var(--radius-sm)] bg-[var(--color-error-500)]/10 border border-[var(--color-error-500)]/30">
            <p className="text-sm text-[var(--color-error-500)]">{error}</p>
          </div>
        )}

        {status?.status === "completed" && (
          <div className="p-3 rounded-[var(--radius-sm)] bg-[var(--color-cobalt-600)]/10 border border-[var(--color-cobalt-600)]/30">
            <p className="text-sm text-[var(--color-cobalt-600)]">Caption generation completed successfully.</p>
          </div>
        )}
      </div>

      <div className="flex justify-between mt-6 pt-4 border-t border-[var(--color-border-subtle)]">
        <div className="flex gap-2">
          <Button variant="primary" onClick={handleGenerate} disabled={isRunning || isPreviewing}>
            {isRunning ? "Generating..." : "Create Captions"}
          </Button>
          <Button variant="secondary" onClick={handlePreview} disabled={isRunning || isPreviewing || !state.folder}>
            {isPreviewing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Eye className="w-4 h-4" />}
            <span className="ml-1">Preview</span>
          </Button>
          {isRunning && (
            <Button variant="danger" onClick={handleCancel}>
              Cancel
            </Button>
          )}
        </div>
        <Button variant="secondary" onClick={onClose} disabled={isRunning}>
          Close
        </Button>
      </div>
    </ModalBase>
  );
}
