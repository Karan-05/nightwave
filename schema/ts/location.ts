/**
 * Location — a geocodable place referenced in evidence,
 * displayed on the investigation map.
 */
export interface Location {
  location_id: string;
  evidence_id: string;
  case_id: string;

  name: string;
  address: string;
  /** Null until geocoded */
  latitude: number | null;
  /** Null until geocoded */
  longitude: number | null;

  /** e.g. "text_analysis", "gps_metadata" */
  extraction_method: string;
  /** 0-100 */
  confidence: number;
  priority: 'low' | 'medium' | 'high' | 'critical';
  /** Investigative context */
  description: string;
  /** e.g. "Residence", "Crime Scene", "Business", "Street/Intersection" */
  subtype: string;

  timestamp: string;
  tags: string[];
  extracted_from: string;
  related_entities: any[];
  related_events: any[];
  created_at: string;
}
