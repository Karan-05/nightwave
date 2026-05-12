import type { EvidenceLink } from './hypothesis';

/**
 * Lead — an actionable investigative thread to pursue,
 * possibly spawned from a hypothesis.
 */
export interface Lead {
  id: string;
  title: string;
  description: string;
  priority: 'critical' | 'high' | 'medium' | 'low';
  status: 'active' | 'archived' | 'verified' | 'dismissed';
  assigned_to?: string;
  deadline?: string;
  linked_entities?: string[];
  linked_events?: string[];
  linked_locations?: string[];
  linked_citations?: EvidenceLink[];
  action_items?: string[];
  source_hypothesis_id?: string;
  investigation_notes?: Array<{ text: string; timestamp: string }>;
  metadata?: Record<string, any>;
  created_at?: string;
  updated_at?: string;
}
