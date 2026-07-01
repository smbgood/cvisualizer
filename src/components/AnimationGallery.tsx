import { useEffect, useMemo, useState } from "react";
import type { AnimationItem } from "../types";

type AnimationGalleryProps = {
  animations: AnimationItem[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onRemove: (id: string) => void;
};

export default function AnimationGallery({ animations, selectedId, onSelect, onRemove }: AnimationGalleryProps) {
  const selectedAnimation =
    (selectedId ? animations.find((item) => item.id === selectedId) : animations[0]) ?? null;
  const [playbackIndex, setPlaybackIndex] = useState(0);

  useEffect(() => {
    setPlaybackIndex(0);
  }, [selectedAnimation?.id]);

  useEffect(() => {
    if (!selectedAnimation || selectedAnimation.frames.length <= 1) {
      return;
    }
    const msPerFrame = (selectedAnimation.durationSeconds * 1000) / selectedAnimation.frames.length;
    const timer = window.setInterval(() => {
      setPlaybackIndex((current) => (current + 1) % selectedAnimation.frames.length);
    }, Math.max(msPerFrame, 16));

    return () => window.clearInterval(timer);
  }, [selectedAnimation]);

  const playbackFrame = useMemo(() => {
    if (!selectedAnimation || selectedAnimation.frames.length === 0) {
      return null;
    }
    const safeIndex = Math.min(playbackIndex, selectedAnimation.frames.length - 1);
    return selectedAnimation.frames[safeIndex];
  }, [playbackIndex, selectedAnimation]);

  return (
    <section className="panel">
      <div className="panel-header">
        <h3>Animations</h3>
        <span className="muted">{animations.length} generated</span>
      </div>
      {animations.length === 0 ? (
        <p className="muted">Create animations from timeline frames to build motion previews.</p>
      ) : (
        <>
          <div className="thumb-grid animation-grid">
            {animations.map((item) => (
              <button
                type="button"
                key={item.id}
                className={`thumb ${selectedAnimation?.id === item.id ? "selected" : ""}`}
                onClick={() => onSelect(item.id)}
              >
                <img src={item.frames[0]?.image} alt={`Animation ${item.id}`} />
                <span>
                  {item.frameCount} frames / {item.durationSeconds.toFixed(1)}s
                </span>
                <span className="muted">From frame {item.sourceFrameIndex}</span>
                <span
                  className="remove"
                  onClick={(event) => {
                    event.stopPropagation();
                    onRemove(item.id);
                  }}
                >
                  x
                </span>
              </button>
            ))}
          </div>
          {selectedAnimation && playbackFrame && (
            <div className="animation-player">
              <img src={playbackFrame.image} alt={`Animation frame ${playbackFrame.index}`} />
              <p className="muted">
                Playing frame {playbackFrame.index}/{selectedAnimation.frames.length} | source frame{" "}
                {selectedAnimation.sourceFrameIndex}
              </p>
              <p className="muted">Prompt: {selectedAnimation.effectivePrompt}</p>
            </div>
          )}
        </>
      )}
    </section>
  );
}
