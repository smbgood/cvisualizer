import type { SavedSession, TimelineFrame } from "../types";

type TimelineScrubberProps = {
  frames: TimelineFrame[];
  selectedFrameIndex: number | null;
  currentSessionId: string | null;
  savedSessions: SavedSession[];
  viewedSessionId: string | null;
  onSelectFrame: (frameIndex: number | null) => void;
  onCreateAnimation: (frame: TimelineFrame) => void;
  creatingAnimationFrameIndex: number | null;
  onLoadSession: (sessionId: string) => void;
  onRefreshSessions: () => void;
};

export default function TimelineScrubber({
  frames,
  selectedFrameIndex,
  currentSessionId,
  savedSessions,
  viewedSessionId,
  onSelectFrame,
  onCreateAnimation,
  creatingAnimationFrameIndex,
  onLoadSession,
  onRefreshSessions,
}: TimelineScrubberProps) {
  const selectedFrame =
    selectedFrameIndex === null ? null : frames.find((frame) => frame.index === selectedFrameIndex) ?? null;
  const latestFrame = frames.length > 0 ? frames[frames.length - 1] : null;
  const sliderValue = selectedFrame?.index ?? latestFrame?.index ?? 0;
  const maxFrame = latestFrame?.index ?? 0;

  return (
    <section className="panel">
      <div className="panel-header">
        <h3>Timeline</h3>
        <span className="muted">
          {frames.length} frames
          {frames.some((frame) => frame.frameKind === "study") ? " incl. study frames" : ""}
        </span>
      </div>

      {frames.length === 0 ? (
        <p className="muted">Generated frames will appear here for timeline scrubbing.</p>
      ) : (
        <>
          <div className="control-row timeline-controls">
            <label>
              Frame {selectedFrame ? selectedFrame.index : maxFrame}
              {selectedFrame?.frameKind === "study" && selectedFrame.studyStep && selectedFrame.studyTotal
                ? ` (study ${selectedFrame.studyStep}/${selectedFrame.studyTotal})`
                : ""}
              <input
                type="range"
                min={frames[0].index}
                max={maxFrame}
                step={1}
                value={sliderValue}
                onChange={(event) => onSelectFrame(Number(event.target.value))}
              />
            </label>
            <button type="button" onClick={() => onSelectFrame(null)}>
              Live
            </button>
          </div>
          <div className="filmstrip">
            {frames.map((frame) => (
              <div key={frame.index} className="filmstrip-item">
                <button
                  type="button"
                  className={`filmstrip-frame ${selectedFrameIndex === frame.index ? "selected" : ""} ${
                    frame.frameKind === "study" ? "study-frame" : ""
                  }`}
                  onClick={() => onSelectFrame(frame.index)}
                >
                  <img src={frame.image} alt={`Frame ${frame.index}`} />
                  <span>
                    {frame.index}
                    {frame.frameKind === "study" ? " study" : ""}
                  </span>
                </button>
                <button
                  type="button"
                  className="filmstrip-action"
                  onClick={() => onCreateAnimation(frame)}
                  disabled={creatingAnimationFrameIndex !== null}
                >
                  {creatingAnimationFrameIndex === frame.index ? "Creating..." : "Create animation"}
                </button>
              </div>
            ))}
          </div>
          {selectedFrame?.effectivePrompt && (
            <p className="muted">Selected effective prompt: {selectedFrame.effectivePrompt}</p>
          )}
        </>
      )}

      <div className="session-row">
        <span className="muted">
          {viewedSessionId
            ? `Viewing saved session ${viewedSessionId}`
            : currentSessionId
              ? `Saving to outputs/${currentSessionId}`
              : "Waiting for backend session..."}
        </span>
        <button type="button" onClick={onRefreshSessions}>
          Refresh Sessions
        </button>
      </div>

      {savedSessions.length > 0 && (
        <div className="saved-session-grid">
          {savedSessions.map((session) => (
            <button
              type="button"
              key={session.id}
              className={`saved-session ${viewedSessionId === session.id ? "selected" : ""}`}
              onClick={() => onLoadSession(session.id)}
            >
              {session.thumbnail_url && <img src={session.thumbnail_url} alt={`Session ${session.id}`} />}
              <span>{new Date(session.created_at).toLocaleString()}</span>
              <span className="muted">
                {session.frame_count} frames, {session.engine}
              </span>
            </button>
          ))}
        </div>
      )}
    </section>
  );
}
