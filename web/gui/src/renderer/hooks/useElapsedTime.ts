import { useEffect, useState } from "react";

export function useElapsedTime(startTime: number | null, active: boolean): number {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!active || !startTime) {
      setElapsed(0);
      return;
    }
    const tick = () => setElapsed(Math.floor((Date.now() - startTime) / 1000));
    tick();
    const interval = setInterval(tick, 1000);
    return () => clearInterval(interval);
  }, [active, startTime]);

  return elapsed;
}
