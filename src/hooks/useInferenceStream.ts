import { useEffect, useRef, useState } from "react";
import type { StreamSettings, TimelineFrame } from "../types";

export const BACKEND_URL = "http://127.0.0.1:8000";
const WS_URL = "ws://127.0.0.1:8000/ws/stream";

type StreamState = {
  connected: boolean;
  latestFrame: string | null;
  sessionId: string | null;
  timelineFrames: TimelineFrame[];
  engineName: string;
  statusMessage: string;
  deltaFromPrevious: number | null;
  stagnantFrames: number;
  variationApplied: boolean;
  variationTriggered: boolean;
  variationPulseRemaining: number;
  effectivePrompt: string | null;
};

export function backendAssetUrl(url: string) {
  return url.startsWith("/") ? `${BACKEND_URL}${url}` : url;
}

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
    sessionId: null,
    timelineFrames: [],
    engineName: "unknown",
    statusMessage: "Checking backend...",
    deltaFromPrevious: null,
    stagnantFrames: 0,
    variationApplied: false,
    variationTriggered: false,
    variationPulseRemaining: 0,
    effectivePrompt: null,
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
          session_id?: string;
          frame_index?: number;
          frame_url?: string;
          created_at?: string;
          engine?: string;
          detail?: string;
          delta_from_previous?: number | null;
          stagnant_frames?: number;
          variation_applied?: boolean;
          variation_triggered?: boolean;
          variation_pulse_remaining?: number;
          effective_prompt?: string;
        };
        if (payload.type === "session" && typeof payload.session_id === "string") {
          setState((current) => ({
            ...current,
            latestFrame: null,
            sessionId: payload.session_id ?? current.sessionId,
            timelineFrames: [],
            deltaFromPrevious: null,
            stagnantFrames: 0,
            variationApplied: false,
            variationTriggered: false,
            variationPulseRemaining: 0,
            effectivePrompt: null,
          }));
        }
        if (payload.type === "frame" && typeof payload.frame === "string") {
          const nextFrame = payload.frame;
          setState((current) => {
            const frameUrl =
              typeof payload.frame_url === "string" ? backendAssetUrl(payload.frame_url) : nextFrame;
            const frameIndex =
              typeof payload.frame_index === "number"
                ? payload.frame_index
                : current.timelineFrames.length + 1;

            return {
              ...current,
              latestFrame: nextFrame,
              sessionId: payload.session_id ?? current.sessionId,
              timelineFrames: [
                ...current.timelineFrames,
                {
                  index: frameIndex,
                  image: frameUrl,
                  createdAt: payload.created_at ?? new Date().toISOString(),
                  deltaFromPrevious:
                    typeof payload.delta_from_previous === "number" ? payload.delta_from_previous : null,
                  stagnantFrames:
                    typeof payload.stagnant_frames === "number" ? payload.stagnant_frames : undefined,
                  variationApplied:
                    typeof payload.variation_applied === "boolean" ? payload.variation_applied : undefined,
                  variationTriggered:
                    typeof payload.variation_triggered === "boolean" ? payload.variation_triggered : undefined,
                  variationPulseRemaining:
                    typeof payload.variation_pulse_remaining === "number"
                      ? payload.variation_pulse_remaining
                      : undefined,
                  effectivePrompt:
                    typeof payload.effective_prompt === "string" ? payload.effective_prompt : undefined,
                },
              ],
              deltaFromPrevious:
                typeof payload.delta_from_previous === "number" ? payload.delta_from_previous : null,
              stagnantFrames: typeof payload.stagnant_frames === "number" ? payload.stagnant_frames : 0,
              variationApplied: Boolean(payload.variation_applied),
              variationTriggered: Boolean(payload.variation_triggered),
              variationPulseRemaining:
                typeof payload.variation_pulse_remaining === "number" ? payload.variation_pulse_remaining : 0,
              effectivePrompt: typeof payload.effective_prompt === "string" ? payload.effective_prompt : null,
            };
          });
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
