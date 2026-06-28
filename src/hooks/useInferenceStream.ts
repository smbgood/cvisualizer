import { useEffect, useRef, useState } from "react";
import type { StreamSettings } from "../types";

const BACKEND_URL = "http://127.0.0.1:8000";
const WS_URL = "ws://127.0.0.1:8000/ws/stream";

type StreamState = {
  connected: boolean;
  latestFrame: string | null;
  engineName: string;
  statusMessage: string;
};

function sendSettings(socket: WebSocket, settings: StreamSettings) {
  socket.send(
    JSON.stringify({
      type: "settings",
      payload: settings,
    }),
  );
}

function sendSeedFrame(socket: WebSocket, seedDataUrl: string) {
  socket.send(
    JSON.stringify({
      type: "seed_frame",
      image: seedDataUrl,
    }),
  );
}

export function useInferenceStream(seedDataUrl: string | null, settings: StreamSettings) {
  const [state, setState] = useState<StreamState>({
    connected: false,
    latestFrame: null,
    engineName: "unknown",
    statusMessage: "Checking backend...",
  });

  const socketRef = useRef<WebSocket | null>(null);
  const latestSeedRef = useRef<string | null>(seedDataUrl);
  const latestSettingsRef = useRef<StreamSettings>(settings);

  useEffect(() => {
    let active = true;

    const syncStatus = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/api/status`);
        if (!response.ok) {
          throw new Error("status endpoint failed");
        }
        const data = (await response.json()) as {
          engine: string;
          model_ready: boolean;
          detail: string;
        };

        if (!active) {
          return;
        }

        setState((current) => ({
          ...current,
          engineName: data.engine,
          statusMessage: data.detail,
        }));
      } catch {
        if (!active) {
          return;
        }
        setState((current) => ({
          ...current,
          statusMessage: "Backend unavailable. Start FastAPI to stream frames.",
        }));
      }
    };

    syncStatus();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    const socket = new WebSocket(WS_URL);
    socketRef.current = socket;

    socket.onopen = () => {
      setState((current) => ({ ...current, connected: true }));
      sendSettings(socket, latestSettingsRef.current);
      if (latestSeedRef.current) {
        sendSeedFrame(socket, latestSeedRef.current);
      }
    };

    socket.onclose = () => {
      setState((current) => ({ ...current, connected: false }));
    };

    socket.onerror = () => {
      setState((current) => ({
        ...current,
        connected: false,
        statusMessage: "Socket error. Check backend logs.",
      }));
    };

    socket.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data) as {
          type?: string;
          frame?: string;
          engine?: string;
          detail?: string;
        };
        if (payload.type === "frame" && typeof payload.frame === "string") {
          const nextFrame = payload.frame;
          setState((current) => ({ ...current, latestFrame: nextFrame }));
        }
        if (payload.engine || payload.detail) {
          setState((current) => ({
            ...current,
            engineName: payload.engine ?? current.engineName,
            statusMessage: payload.detail ?? current.statusMessage,
          }));
        }
      } catch {
        // Ignore malformed payloads and keep streaming.
      }
    };

    return () => {
      socket.close();
      socketRef.current = null;
    };
  }, []);

  useEffect(() => {
    latestSettingsRef.current = settings;
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      return;
    }
    sendSettings(socket, settings);
  }, [settings]);

  useEffect(() => {
    latestSeedRef.current = seedDataUrl;
    const socket = socketRef.current;
    if (!seedDataUrl || !socket || socket.readyState !== WebSocket.OPEN) {
      return;
    }
    sendSeedFrame(socket, seedDataUrl);
  }, [seedDataUrl]);

  return state;
}
