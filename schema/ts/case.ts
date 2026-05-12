/**
 * Case — the top-level investigation container.
 */
export interface Case {
  case_id: string;
  case_name: string;
  summary: string;
  status: 'open' | 'closed';
  priority: 'high' | 'medium' | 'low' | 'critical';
  assigned_to?: string;
  tags: string[];
  evidence_count: number;
  created_at: string;
  updated_at: string;
}
