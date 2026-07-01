export type StreamSettings = {
  prompt: string;
  strength: number;
  running: boolean;
  anti_stagnation_enabled: boolean;
  stagnation_threshold: number;
  stagnation_window: number;
  variation_strength: number;
  prompt_enhancement_enabled: boolean;
  prompt_enhancement_interval: number;
  prompt_enhancement_strength: number;
  study_frames_enabled: boolean;
  study_frame_count: number;
  study_frame_strength: number;
  study_frame_effort: number;
  study_frame_delay: number;
};

export type TimelineFrame = {
  index: number;
  image: string;
  frameKind?: "generated" | "study";
  createdAt: string;
  generationIndex?: number;
  studyStep?: number;
  studyTotal?: number;
  studyFrameEffort?: number;
  seedIndex?: number | null;
  seedUrl?: string | null;
  prompt?: string;
  strength?: number;
  deltaFromPrevious?: number | null;
  stagnantFrames?: number;
  variationApplied?: boolean;
  variationTriggered?: boolean;
  variationPulseRemaining?: number;
  promptEnhancementEnabled?: boolean;
  promptEnhancementInterval?: number;
  promptEnhancementStrength?: number;
  promptEnhancementRefreshed?: boolean;
  enhancedPrompt?: string | null;
  promptEnhancementLastFrame?: number;
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
