import type { Entity } from './entity';

/**
 * EntityRelationship — a directed edge between two entities
 * on the investigation graph.
 */
export interface EntityRelationship {
  /** Legacy alias for relationship_id */
  id?: string;
  relationship_id: string;
  case_id?: string;
  evidence_id?: string;

  /** Source entity name (alternative to source_entity_id) */
  source?: string;
  source_name?: string;
  /** Target entity name (alternative to target_entity_id) */
  target?: string;
  target_name?: string;
  source_entity_id?: string;
  target_entity_id?: string;

  relationship_type?: string;
  description: string;
  /** Relationship strength */
  strength?: number;
  metadata?: Record<string, any>;
  properties?: Record<string, any>;
  created_at: string;
  updated_at?: string | null;

  source_entity?: Entity;
  target_entity?: Entity;

  /** Concise label for graph edges */
  short_description: string;
  /** 0-100 */
  confidence: number;
  priority: 'low' | 'medium' | 'high' | 'critical';
  /** Detailed explanation */
  long_description?: string;

  // Cross-evidence tracking
  /** JSON string of evidence IDs */
  evidence_ids?: string;
  annotations_count?: number;
}
