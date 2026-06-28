export type SnapshotItem = {
  id: string;
  image: string;
  createdAt: number;
};

type SnapshotGalleryProps = {
  snapshots: SnapshotItem[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onRemove: (id: string) => void;
};

export default function SnapshotGallery({
  snapshots,
  selectedId,
  onSelect,
  onRemove,
}: SnapshotGalleryProps) {
  return (
    <section className="panel">
      <div className="panel-header">
        <h3>Snapshots</h3>
        <span className="muted">{snapshots.length} captured</span>
      </div>
      {snapshots.length === 0 ? (
        <p className="muted">Capture frames to build a visual trail.</p>
      ) : (
        <div className="thumb-grid">
          {snapshots.map((item) => (
            <button
              type="button"
              key={item.id}
              className={`thumb ${selectedId === item.id ? "selected" : ""}`}
              onClick={() => onSelect(item.id)}
            >
              <img src={item.image} alt={`Snapshot ${item.id}`} />
              <span>{new Date(item.createdAt).toLocaleTimeString()}</span>
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
      )}
    </section>
  );
}
