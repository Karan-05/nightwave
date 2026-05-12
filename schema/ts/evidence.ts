/**
 * Evidence — a file (document, video, audio, image, etc.)
 * uploaded to a case for analysis.
 */
export interface Evidence {
  /** Legacy alias for evidence_id */
  id?: string;
  evidence_id: string;
  case_id: string;
  /** Legacy aliases used by older UI modules */
  file_name?: string;
  filename?: string;
  title?: string;
  type?: string;
  file_type?: string;
  evidence_name: string;
  evidence_type: string;
  file_path?: string;
  description: string;
  metadata: Record<string, any>;
  uploaded_at: string;
  file_size?: number;
  mime_type?: string;

  // Processing state
  processing_status?: 'pending' | 'processing' | 'completed' | 'failed';
  processing_progress?: number;
  processing_stage?: string;
  processing_message?: string;

  // Extraction results
  content_summary?: string;
  extracted_entities?: string[];
  extracted_locations?: string[];
  confidence_score?: number;

  // Media-specific
  /** Video/audio duration in seconds */
  duration?: number;
  /** Document page count */
  pages?: number;

  // Classification / quality
  tags?: string[];
  /** LLM-generated summary */
  ai_summary?: string;
  /** Quality score 0-100 */
  confidence?: number;
  priority?: 'low' | 'medium' | 'high' | 'critical';
  status?: string;
}
