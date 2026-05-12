/**
 * EvidenceLink — a pointer from a hypothesis to a specific
 * excerpt / page / timestamp within an evidence file.
 */
export interface EvidenceLink {
  id: string;
  evidence_path: string;
  excerpt?: string;
  weight?: number;
  confidence?: number;
  page?: number;
  line?: number;
  timestamp?: number;
  notes?: string;
}

/**
 * Hypothesis — a falsifiable claim about the case,
 * supported or refuted by evidence links.
 */
export interface Hypothesis {
  id: string;
  statement: string;
  /** 0-1 confidence score */
  confidence: number;
  status: 'active' | 'confirmed' | 'refuted' | 'superseded';
  supporting_evidence: EvidenceLink[];
  refuting_evidence: EvidenceLink[];
  reasoning?: string;
  assumptions?: string[];
  implications?: string[];
  generated_leads?: string[];
  alternative_hypotheses?: string[];
  resolution?: string;
  resolution_reasoning?: string;
  metadata?: Record<string, any>;
  created_at?: string;
  updated_at?: string;
}
