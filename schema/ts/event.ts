/**
 * Event — a discrete occurrence on the investigation timeline,
 * extracted from evidence or created by an analyst.
 */
export interface Event {
  /** Legacy alias for event_id */
  id?: string;
  event_id: string;
  case_id?: string;
  evidence_id?: string;

  /** Display name (preferred over event_name) */
  name?: string;
  event_name?: string;

  event_type: string;
  description: string;

  // Temporal positioning
  timestamp?: string;
  timestamp_type?: string;
  timestamp_absolute?: string;
  timestamp_relative_to?: string | null;
  timestamp_relative_offset?: number | null;
  temporal_order?: number;
  temporal_group?: string | null;

  // Linked artifacts (names / IDs)
  entities?: string[];
  location?: string;

  metadata?: Record<string, any>;
  properties?: Record<string, any>;
  created_at: string;
  updated_at?: string | null;

  // Classification
  priority?: 'low' | 'medium' | 'high' | 'critical';
  /** Seconds or formatted string */
  duration?: string | number;
  /** e.g. "COMMUNICATION", "MOVEMENT", "TRANSACTION" */
  category?: string;
  verified?: boolean;
  confidence?: number;

  // Relationship arrays
  people_involved?: string[];
  objects_involved?: string[];
  locations?: string[];

  // Metadata fields
  connections_count?: number;
  source?: string;
  /** 0-10 */
  reliability?: number;
  cross_reference?: string[];
  source_timestamps?: Array<{
    evidence_id: string;
    evidence_name: string;
    start: number;
    end: number;
  }>;
}
