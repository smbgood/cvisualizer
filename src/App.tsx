import { useEffect, useMemo, useState } from "react";
import LiveDisplay from "./components/LiveDisplay";
import SeedCanvas from "./components/SeedCanvas";
import SnapshotGallery, { type SnapshotItem } from "./components/SnapshotGallery";
import TimelineScrubber from "./components/TimelineScrubber";
import { BACKEND_URL, backendAssetUrl, useInferenceStream } from "./hooks/useInferenceStream";
import type { SavedSession, StreamSettings, TimelineFrame } from "./types";

function createSnapshot(frame: string): SnapshotItem {
  return {
    id: crypto.randomUUID(),
    image: frame,
    createdAt: Date.now(),
  };
}

async function imageUrlToDataUrl(imageUrl: string): Promise<string | null> {
  try {
    const response = await fetch(imageUrl);
    if (!response.ok) {
      return null;
    }
    const blob = await response.blob();
    return await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(typeof reader.result === "string" ? reader.result : null);
      reader.onerror = () => reject(new Error(`Failed to read seed image: ${imageUrl}`));
      reader.readAsDataURL(blob);
    });
  } catch {
    return null;
  }
}

export default function App() {
  const [seedDataUrl, setSeedDataUrl] = useState<string | null>(null);
  const [prompt, setPrompt] = useState("radical surreal neon transformation");
  const [strength, setStrength] = useState(0.85);
  const [running, setRunning] = useState(true);
  const [antiStagnationEnabled, setAntiStagnationEnabled] = useState(true);
  const [stagnationThreshold, setStagnationThreshold] = useState(0.012);
  const [stagnationWindow, setStagnationWindow] = useState(6);
  const [variationStrength, setVariationStrength] = useState(0.35);
  const [snapshots, setSnapshots] = useState<SnapshotItem[]>([]);
  const [selectedSnapshotId, setSelectedSnapshotId] = useState<string | null>(null);
  const [selectedFrameIndex, setSelectedFrameIndex] = useState<number | null>(null);
  const [savedSessions, setSavedSessions] = useState<SavedSession[]>([]);
  const [viewedTimeline, setViewedTimeline] = useState<{ id: string; frames: TimelineFrame[] } | null>(null);

  const settings: StreamSettings = useMemo(
    () => ({
      prompt,
      strength,
      running,
      anti_stagnation_enabled: antiStagnationEnabled,
      stagnation_threshold: stagnationThreshold,
      stagnation_window: stagnationWindow,
      variation_strength: variationStrength,
    }),
    [prompt, strength, running, antiStagnationEnabled, stagnationThreshold, stagnationWindow, variationStrength],
  );

  const stream = useInferenceStream(seedDataUrl, settings);
  const timelineFrames = viewedTimeline?.frames ?? stream.timelineFrames;
  const selectedTimelineFrame =
    selectedFrameIndex === null
      ? null
      : timelineFrames.find((frame) => frame.index === selectedFrameIndex)?.image ?? null;
  const displayImage =
    snapshots.find((item) => item.id === selectedSnapshotId)?.image ?? selectedTimelineFrame ?? stream.latestFrame;

  const refreshSavedSessions = async () => {
    const response = await fetch(`${BACKEND_URL}/api/sessions`);
    if (!response.ok) {
      return;
    }

    const data = (await response.json()) as { sessions: SavedSession[] };
    setSavedSessions(
      data.sessions.map((session) => ({
        ...session,
        thumbnail_url: session.thumbnail_url ? backendAssetUrl(session.thumbnail_url) : session.thumbnail_url,
      })),
    );
  };

  useEffect(() => {
    void refreshSavedSessions();
  }, []);

  const loadSavedSession = async (sessionId: string) => {
    const response = await fetch(`${BACKEND_URL}/api/sessions/${sessionId}`);
    if (!response.ok) {
      return;
    }

    const data = (await response.json()) as {
      frames: Array<{
        index: number;
        url: string;
        created_at: string;
        seed_index?: number | null;
        seed_url?: string | null;
        prompt?: string;
        strength?: number;
        delta_from_previous?: number | null;
        stagnant_frames?: number;
        variation_applied?: boolean;
        variation_triggered?: boolean;
        variation_pulse_remaining?: number;
        effective_prompt?: string;
      }>;
    };
    const frames = data.frames.map((frame) => ({
      index: frame.index,
      image: backendAssetUrl(frame.url),
      createdAt: frame.created_at,
      seedIndex: frame.seed_index ?? null,
      seedUrl: frame.seed_url ? backendAssetUrl(frame.seed_url) : null,
      prompt: frame.prompt,
      strength: frame.strength,
      deltaFromPrevious: frame.delta_from_previous ?? null,
      stagnantFrames: frame.stagnant_frames,
      variationApplied: frame.variation_applied,
      variationTriggered: frame.variation_triggered,
      variationPulseRemaining: frame.variation_pulse_remaining,
      effectivePrompt: frame.effective_prompt,
    }));

    setViewedTimeline({ id: sessionId, frames });
    setSelectedSnapshotId(null);
    setSelectedFrameIndex(frames[frames.length - 1]?.index ?? null);
  };

  const selectTimelineFrame = (frameIndex: number | null) => {
    setSelectedSnapshotId(null);
    setSelectedFrameIndex(frameIndex);
    if (frameIndex === null) {
      setViewedTimeline(null);
    }
  };

  useEffect(() => {
    if (!viewedTimeline || selectedFrameIndex === null) {
      return;
    }
    const frame = timelineFrames.find((item) => item.index === selectedFrameIndex);
    const seedUrl = frame?.seedUrl;
    if (!seedUrl) {
      return;
    }

    let cancelled = false;
    const restoreSeed = async () => {
      const restoredSeed = await imageUrlToDataUrl(seedUrl);
      if (!cancelled && restoredSeed) {
        setSeedDataUrl(restoredSeed);
      }
    };

    void restoreSeed();
    return () => {
      cancelled = true;
    };
  }, [selectedFrameIndex, timelineFrames, viewedTimeline]);

  const captureSnapshot = () => {
    if (!displayImage) {
      return;
    }

    const next = [createSnapshot(displayImage), ...snapshots];
    setSnapshots(next);
    setSelectedSnapshotId(next[0].id);
  };

  const removeSnapshot = (id: string) => {
    const next = snapshots.filter((item) => item.id !== id);
    setSnapshots(next);
    if (selectedSnapshotId === id) {
      setSelectedSnapshotId(next[0]?.id ?? null);
    }
  };

  return (
    <main className="app">
      <header className="app-header">
        <h1>cvisualizer</h1>
        <p>Sketch or import a seed image, stream AI transformations, then capture visual milestones.</p>
      </header>

      <section className="grid-top">
        <SeedCanvas onSeedChange={setSeedDataUrl} seedDataUrl={seedDataUrl} />
        <div className="stack">
          <LiveDisplay
            frameDataUrl={displayImage}
            connected={stream.connected}
            engineName={stream.engineName}
            statusMessage={stream.statusMessage}
          />
          <section className="panel">
            <h3>Controls</h3>
            <div className="control-row">
              <label>
                Prompt
                <input value={prompt} onChange={(event) => setPrompt(event.target.value)} />
              </label>
            </div>
            <div className="control-row">
              <label>
                Strength {strength.toFixed(2)}
                <input
                  type="range"
                  min={0.05}
                  max={1}
                  step={0.01}
                  value={strength}
                  onChange={(event) => setStrength(Number(event.target.value))}
                />
              </label>
            </div>
            <div className="control-row">
              <label className="checkbox-row">
                <input
                  type="checkbox"
                  checked={antiStagnationEnabled}
                  onChange={(event) => setAntiStagnationEnabled(event.target.checked)}
                />
                <span>Enable anti-stagnation</span>
              </label>
            </div>
            <div className="control-row">
              <label>
                Stagnation threshold {stagnationThreshold.toFixed(4)}
                <input
                  type="range"
                  min={0.001}
                  max={0.04}
                  step={0.001}
                  value={stagnationThreshold}
                  onChange={(event) => setStagnationThreshold(Number(event.target.value))}
                  disabled={!antiStagnationEnabled}
                />
              </label>
            </div>
            <div className="control-row">
              <label>
                Detection window {stagnationWindow} frames
                <input
                  type="range"
                  min={2}
                  max={20}
                  step={1}
                  value={stagnationWindow}
                  onChange={(event) => setStagnationWindow(Number(event.target.value))}
                  disabled={!antiStagnationEnabled}
                />
              </label>
            </div>
            <div className="control-row">
              <label>
                Variation strength {variationStrength.toFixed(2)}
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={variationStrength}
                  onChange={(event) => setVariationStrength(Number(event.target.value))}
                  disabled={!antiStagnationEnabled}
                />
              </label>
            </div>
            <p className="muted">
              Delta: {stream.deltaFromPrevious === null ? "n/a" : stream.deltaFromPrevious.toFixed(4)} | stagnant:{" "}
              {stream.stagnantFrames} | pulse: {stream.variationPulseRemaining}{" "}
              {stream.variationApplied ? "(variation active)" : ""}
            </p>
            <div className="control-row">
              <button type="button" onClick={() => setRunning((value) => !value)}>
                {running ? "Pause Generation" : "Resume Generation"}
              </button>
              <button type="button" onClick={captureSnapshot} disabled={!displayImage}>
                Capture Snapshot
              </button>
            </div>
          </section>
        </div>
      </section>

      <TimelineScrubber
        frames={timelineFrames}
        selectedFrameIndex={selectedFrameIndex}
        currentSessionId={stream.sessionId}
        savedSessions={savedSessions}
        viewedSessionId={viewedTimeline?.id ?? null}
        onSelectFrame={selectTimelineFrame}
        onLoadSession={loadSavedSession}
        onRefreshSessions={refreshSavedSessions}
      />

      <SnapshotGallery
        snapshots={snapshots}
        selectedId={selectedSnapshotId}
        onSelect={setSelectedSnapshotId}
        onRemove={removeSnapshot}
      />
    </main>
  );
}
