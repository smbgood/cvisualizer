import { useEffect, useRef, useState } from "react";

type SeedCanvasProps = {
  onSeedChange: (dataUrl: string) => void;
};

const CANVAS_SIZE = 320;

function drawChecker(context: CanvasRenderingContext2D) {
  const block = 20;
  for (let y = 0; y < CANVAS_SIZE; y += block) {
    for (let x = 0; x < CANVAS_SIZE; x += block) {
      context.fillStyle = (x / block + y / block) % 2 === 0 ? "#20273b" : "#1a2030";
      context.fillRect(x, y, block, block);
    }
  }
}

export default function SeedCanvas({ onSeedChange }: SeedCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [brushSize, setBrushSize] = useState(14);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }

    context.imageSmoothingEnabled = false;
    drawChecker(context);
    onSeedChange(canvas.toDataURL("image/png"));
  }, [onSeedChange]);

  const paint = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !canvasRef.current) {
      return;
    }

    const rect = canvasRef.current.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    const context = canvasRef.current.getContext("2d");
    if (!context) {
      return;
    }

    context.fillStyle = "#f2f5ff";
    context.beginPath();
    context.arc(x, y, brushSize, 0, Math.PI * 2);
    context.fill();
  };

  const finishStroke = () => {
    if (!canvasRef.current) {
      return;
    }

    setIsDrawing(false);
    onSeedChange(canvasRef.current.toDataURL("image/png"));
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }
    drawChecker(context);
    onSeedChange(canvas.toDataURL("image/png"));
  };

  const importImage = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file || !canvasRef.current) {
      return;
    }
    const imageUrl = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      const context = canvasRef.current?.getContext("2d");
      if (!context || !canvasRef.current) {
        URL.revokeObjectURL(imageUrl);
        return;
      }

      context.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
      context.drawImage(image, 0, 0, CANVAS_SIZE, CANVAS_SIZE);
      onSeedChange(canvasRef.current.toDataURL("image/png"));
      URL.revokeObjectURL(imageUrl);
    };
    image.src = imageUrl;
  };

  return (
    <section className="panel">
      <h3>Seed Input</h3>
      <canvas
        ref={canvasRef}
        width={CANVAS_SIZE}
        height={CANVAS_SIZE}
        className="seed-canvas"
        onPointerDown={() => setIsDrawing(true)}
        onPointerUp={finishStroke}
        onPointerLeave={finishStroke}
        onPointerMove={paint}
      />
      <div className="control-row">
        <label>
          Brush
          <input
            type="range"
            min={2}
            max={40}
            value={brushSize}
            onChange={(event) => setBrushSize(Number(event.target.value))}
          />
        </label>
      </div>
      <div className="control-row">
        <button type="button" onClick={clearCanvas}>
          Clear
        </button>
        <label className="button-like">
          Import Image
          <input type="file" accept="image/*" onChange={importImage} />
        </label>
      </div>
    </section>
  );
}
