import { Activity, Database, Image, RefreshCw, Video, Wand2 } from "lucide-react";
import type { ReactNode } from "react";
import { useState } from "react";

import { CaptionToolModal } from "@/components/modals/CaptionToolModal";
import { ConvertModelModal } from "@/components/modals/ConvertModelModal";
import { MaskToolModal } from "@/components/modals/MaskToolModal";
import { ProfilingPanel } from "@/components/modals/ProfilingPanel";
import { StandaloneSamplingModal } from "@/components/modals/StandaloneSamplingModal";
import { VideoToolModal } from "@/components/modals/VideoToolModal";
import { Button, Card } from "@/components/shared";

interface ToolCardProps {
  icon: ReactNode;
  title: string;
  description: string;
  onLaunch?: () => void;
}

function ToolCard({ icon, title, description, onLaunch }: ToolCardProps) {
  return (
    <Card className="flex flex-col items-center text-center gap-4">
      <div className="w-12 h-12 rounded-full flex items-center justify-center bg-[var(--color-border-subtle)]">
        {icon}
      </div>
      <h3 className="text-base font-semibold text-[var(--color-on-surface)]">{title}</h3>
      <p className="text-sm text-[var(--color-on-surface-secondary)]">{description}</p>
      <Button variant="secondary" disabled={!onLaunch} onClick={onLaunch}>
        Launch
      </Button>
    </Card>
  );
}

export default function ToolsPage() {
  const [profilingOpen, setProfilingOpen] = useState(false);
  const [captionOpen, setCaptionOpen] = useState(false);
  const [maskOpen, setMaskOpen] = useState(false);
  const [videoOpen, setVideoOpen] = useState(false);
  const [convertOpen, setConvertOpen] = useState(false);
  const [samplingOpen, setSamplingOpen] = useState(false);

  return (
    <>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        <ToolCard
          icon={<Database className="w-6 h-6 text-[var(--color-cobalt-600)]" />}
          title="Generate Captions"
          description="Batch generate captions using BLIP, WD14, OpenAI, or Gemini API"
          onLaunch={() => setCaptionOpen(true)}
        />
        <ToolCard
          icon={<Wand2 className="w-6 h-6 text-[var(--color-cobalt-600)]" />}
          title="Generate Masks"
          description="Batch generate or interactively edit masks with YOLO support"
          onLaunch={() => setMaskOpen(true)}
        />
        <ToolCard
          icon={<Video className="w-6 h-6 text-[var(--color-cobalt-600)]" />}
          title="Video Tools"
          description="Extract clips, images, and download videos"
          onLaunch={() => setVideoOpen(true)}
        />
        <ToolCard
          icon={<RefreshCw className="w-6 h-6 text-[var(--color-cobalt-600)]" />}
          title="Convert Model"
          description="Convert between model formats"
          onLaunch={() => setConvertOpen(true)}
        />
        <ToolCard
          icon={<Image className="w-6 h-6 text-[var(--color-cobalt-600)]" />}
          title="Sampling"
          description="Generate samples from a trained model"
          onLaunch={() => setSamplingOpen(true)}
        />
        <ToolCard
          icon={<Activity className="w-6 h-6 text-[var(--color-cobalt-600)]" />}
          title="Profiling"
          description="Profile training performance"
          onLaunch={() => setProfilingOpen(true)}
        />
      </div>

      <ProfilingPanel open={profilingOpen} onClose={() => setProfilingOpen(false)} />
      <CaptionToolModal open={captionOpen} onClose={() => setCaptionOpen(false)} />
      <MaskToolModal open={maskOpen} onClose={() => setMaskOpen(false)} />
      <VideoToolModal open={videoOpen} onClose={() => setVideoOpen(false)} />
      <ConvertModelModal open={convertOpen} onClose={() => setConvertOpen(false)} />
      <StandaloneSamplingModal open={samplingOpen} onClose={() => setSamplingOpen(false)} />
    </>
  );
}
