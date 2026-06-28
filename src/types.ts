export type StreamSettings = {
  prompt: string;
  strength: number;
  running: boolean;
  anti_stagnation_enabled: boolean;
  stagnation_threshold: number;
  stagnation_window: number;
  variation_strength: number;
};

export type TimelineFrame = {
  index: number;
  image: string;
  createdAt: string;
  prompt?: string;
  strength?: number;
  deltaFromPrevious?: number | null;
  stagnantFrames?: number;
  variationApplied?: boolean;
  variationTriggered?: boolean;
  variationPulseRemaining?: number;
  effectivePrompt?: string;
};

export type SavedSession = {
  id: string;
  created_at: string;
  ended_at?: string | null;
  engine: string;
  frame_count: number;
  thumbnail_url?: string | null;
};
