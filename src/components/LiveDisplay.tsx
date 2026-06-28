type LiveDisplayProps = {
  frameDataUrl: string | null;
  connected: boolean;
  engineName: string;
  statusMessage: string;
};

export default function LiveDisplay({
  frameDataUrl,
  connected,
  engineName,
  statusMessage,
}: LiveDisplayProps) {
  return (
    <section className="panel">
      <div className="panel-header">
        <h3>Main Display</h3>
        <div className="chip-row">
          <span className={`chip ${connected ? "ok" : "warn"}`}>
            {connected ? "Connected" : "Disconnected"}
          </span>
          <span className="chip">{engineName}</span>
        </div>
      </div>
      <div className="display-frame">
        {frameDataUrl ? <img src={frameDataUrl} alt="Live AI visualization" /> : <p>No frame yet.</p>}
      </div>
      <p className="muted">{statusMessage}</p>
    </section>
  );
}
