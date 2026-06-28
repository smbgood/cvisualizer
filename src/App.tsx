import { useMemo, useState } from "react";
import LiveDisplay from "./components/LiveDisplay";
import SeedCanvas from "./components/SeedCanvas";
import SnapshotGallery, { type SnapshotItem } from "./components/SnapshotGallery";
import { useInferenceStream } from "./hooks/useInferenceStream";
import type { StreamSettings } from "./types";

function createSnapshot(frame: string): SnapshotItem {
  return {
    id: crypto.randomUUID(),
    image: frame,
    createdAt: Date.now(),
  };
}

export default function App() {
  const [seedDataUrl, setSeedDataUrl] = useState<string | null>(null);
  const [prompt, setPrompt] = useState("radical surreal neon transformation");
  const [strength, setStrength] = useState(0.85);
  const [running, setRunning] = useState(true);
  const [snapshots, setSnapshots] = useState<SnapshotItem[]>([]);
  const [selectedSnapshotId, setSelectedSnapshotId] = useState<string | null>(null);

  const settings: StreamSettings = useMemo(
    () => ({
      prompt,
      strength,
      running,
    }),
    [prompt, strength, running],
  );

  const stream = useInferenceStream(seedDataUrl, settings);
  const displayImage =
    snapshots.find((item) => item.id === selectedSnapshotId)?.image ?? stream.latestFrame;

  const captureSnapshot = () => {
    if (!stream.latestFrame) {
      return;
    }

    const next = [createSnapshot(stream.latestFrame), ...snapshots];
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
        <SeedCanvas onSeedChange={setSeedDataUrl} />
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
              <button type="button" onClick={() => setRunning((value) => !value)}>
                {running ? "Pause Generation" : "Resume Generation"}
              </button>
              <button type="button" onClick={captureSnapshot} disabled={!stream.latestFrame}>
                Capture Snapshot
              </button>
            </div>
          </section>
        </div>
      </section>

      <SnapshotGallery
        snapshots={snapshots}
        selectedId={selectedSnapshotId}
        onSelect={setSelectedSnapshotId}
        onRemove={removeSnapshot}
      />
    </main>
  );
}
