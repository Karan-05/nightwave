/**
 * Entity — a person, organization, vehicle, object, or other noun
 * extracted from case evidence.
 */
export interface Entity {
  /** Legacy alias for entity_id */
  id?: string;
  entity_id: string;
  case_id?: string;
  evidence_id?: string;

  /** Display name (preferred over entity_name) */
  name?: string;
  entity_name?: string;

  /** Canonical type string (preferred over entity_type) */
  type?: string;
  entity_type?: string;

  description: string;
  metadata?: Record<string, any>;
  created_at: string;
  updated_at?: string | null;

  /** One-line summary (1-7 words) */
  short_description: string;
  long_description?: string;
  /** Confidence score 0-100 */
  confidence: number;
  priority: 'low' | 'medium' | 'high' | 'critical';

  // Type-specific fields
  /** PERSON only */
  role?: 'Suspect' | 'Victim' | 'Witness' | 'Officer' | 'Driver' | null;
  /** PERSON only, 0-10 */
  threat_level?: number;
  /** VEHICLE only */
  plate?: string | null;
  /** LOCATION only */
  subtype?: string | null;
  /** OBJECT only */
  object_type?: string | null;

  // Metadata / cross-evidence fields
  connections_count?: number;
  status?: string | null;
  /** JSON string containing evidence_ids for cross-evidence tracking */
  properties?: string;
  source_timestamps?: Array<{
    evidence_id: string;
    evidence_name: string;
    start: number;
    end: number;
  }>;
  evidence_ids?: Record<string, number[]>;
  embedding_id?: string;
}
