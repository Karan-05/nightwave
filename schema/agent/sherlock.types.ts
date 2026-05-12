/**
 * Sherlock AI Detective Partner - TypeScript Types
 * 
 * These types match the backend SSE event format exactly as specified
 * in SHERLOCK_FRONTEND_INTEGRATION.md
 */

// =============================================================================
// EVENT TYPES
// =============================================================================

export type SherlockEventType =
    | 'sherlock_start'
    | 'sherlock_thinking'
    | 'sherlock_session_created'
    | 'sherlock_session_resumed'
    | 'sherlock_tool_call'
    | 'sherlock_tool_result'
    | 'sherlock_data_changed'
    | 'sherlock_ui_command'
    | 'sherlock_text'
    | 'sherlock_complete'
    | 'sherlock_error'
    | 'artifact_created'
    | 'artifact_updated'
    | 'sherlock_ui_action';

// =============================================================================
// BASE EVENT
// =============================================================================

export interface SherlockEvent {
    type: SherlockEventType;
    timestamp: string;
}

// =============================================================================
// SPECIFIC EVENT TYPES
// =============================================================================

/** Investigation has started */
export interface SherlockStartEvent extends SherlockEvent {
    type: 'sherlock_start';
    case_id: string;
    query: string;
    mode: 'investigate';
}

/** Agent is reasoning (not tool-related) */
export interface SherlockThinkingEvent extends SherlockEvent {
    type: 'sherlock_thinking';
    message: string;
}

/** New session created with E2B sandbox */
export interface SherlockSessionCreatedEvent extends SherlockEvent {
    type: 'sherlock_session_created';
    session_id: string;
    sandbox_id: string;
    mode: 'investigate';
}

/** Existing session resumed */
export interface SherlockSessionResumedEvent extends SherlockEvent {
    type: 'sherlock_session_resumed';
    session_id: string;
    sandbox_id: string;
    mode: 'investigate';
}

/** Agent is calling a tool - CRITICAL: Store by tool_call_id for correlation */
export interface SherlockToolCallEvent extends SherlockEvent {
    type: 'sherlock_tool_call';
    tool_call_id: string;
    tool: string;
    description: string;
    parameters?: Record<string, any>;
}

/** Tool execution completed - CRITICAL: Match by tool_call_id */
export interface SherlockToolResultEvent extends SherlockEvent {
    type: 'sherlock_tool_result';
    tool_call_id: string;
    tool: string;
    success: boolean;
    output: string;
}

/** Agent speaking directly to the user - Main conversational output */
export interface SherlockTextEvent extends SherlockEvent {
    type: 'sherlock_text';
    text: string;
}

/** Investigation finished */
export interface SherlockCompleteEvent extends SherlockEvent {
    type: 'sherlock_complete';
    session_id: string;
    mode: 'investigate';
    duration_seconds: number;
}

/** An error occurred */
export interface SherlockErrorEvent extends SherlockEvent {
    type: 'sherlock_error';
    error: string;
}

// =============================================================================
// ZERO-AGENT DATA CHANGED EVENT
// =============================================================================

/** Emitted after any CRUD tool succeeds (entity, relationship, event, location, lead, hypothesis, citation) */
export type DataChangedTool = 'entity' | 'relationship' | 'event' | 'location' | 'lead' | 'hypothesis' | 'citation';
export type DataChangedAction = 'create' | 'update' | 'delete' | 'close' | 'add_evidence' | 'resolve';

export interface SherlockDataChangedEvent extends SherlockEvent {
    type: 'sherlock_data_changed';
    tool: DataChangedTool;
    data: {
        action: DataChangedAction;
        [key: string]: any; // entity, relationship, event, location, lead, hypothesis, or citation object
    };
}

// =============================================================================
// ZERO-AGENT UI COMMAND EVENT
// =============================================================================

/** Emitted from the ui_action tool. Lets the agent control the frontend UI. */
export type UICommandAction =
    | 'navigate_tab'
    | 'open_command_rail'
    | 'open_inspector'
    | 'open_evidence'
    | 'highlight_entity'
    | 'highlight_relationship'
    | 'highlight_event'
    | 'highlight_location'
    | 'fit_all_locations'
    | 'toggle_satellite';

export interface SherlockUICommandEvent extends SherlockEvent {
    type: 'sherlock_ui_command';
    ui_action: true;
    action: UICommandAction;
    target?: Record<string, any>;
}

// =============================================================================
// NEW SSE EVENT TYPES (artifact_created, artifact_updated, sherlock_ui_action)
// =============================================================================

/** New SSE format: artifact created by Zero agent */
export interface ArtifactCreatedEvent extends SherlockEvent {
    type: 'artifact_created';
    artifact_type: DataChangedTool;
    artifact: Record<string, any>;
    session_id: string;
}

/** New SSE format: artifact updated by Zero agent */
export interface ArtifactUpdatedEvent extends SherlockEvent {
    type: 'artifact_updated';
    artifact_type: DataChangedTool;
    artifact: Record<string, any>;
    session_id: string;
}

/** 22-member UI action type union */
export type UIActionType =
    // Legacy actions (still supported for backward compat)
    | 'navigate_tab'
    | 'open_command_rail'
    | 'open_inspector'
    | 'highlight_entity'
    | 'highlight_relationship'
    | 'highlight_event'
    | 'highlight_location'
    | 'fit_all_locations'
    | 'toggle_satellite'
    | 'open_lead'
    | 'open_hypothesis'
    | 'navigate_to_leads'
    | 'navigate_to_hypotheses'
    | 'navigate_to_graph'
    | 'navigate_to_timeline'
    | 'navigate_to_map'
    | 'compare_entities'
    | 'suggest_followup'
    // Backend ui_actions.py (EXACT match)
    | 'switch_tab'
    | 'switch_panel'
    | 'close_panel'
    | 'focus_artifact'
    | 'focus_entity'
    | 'focus_event'
    | 'focus_location'
    | 'open_evidence'
    | 'graph_add_entity'
    | 'graph_remove_entity'
    | 'graph_expand_connections'
    | 'graph_filter'
    | 'graph_clear'
    | 'graph_fit_view'
    | 'search'
    | 'filter'
    | 'clear_filters'
    | 'map_fit_all'
    | 'map_zoom_to'
    | 'map_toggle_satellite'
    | 'map_filter'
    | 'map_clear_filters'
    | 'panel_filter'
    | 'panel_clear_filters'
    | 'panel_search'
    | 'timeline_filter'
    | 'evidence_filter';

/** New SSE format: UI action triggered by Zero agent */
export interface UIActionEvent extends SherlockEvent {
    type: 'sherlock_ui_action';
    action: UIActionType;
    target?: Record<string, any>;
    context?: Record<string, any>;
    reasoning: string;
    session_id: string;
}

// =============================================================================
// LEAD & HYPOTHESIS INTERFACES
// =============================================================================

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

export interface Hypothesis {
    id: string;
    statement: string;
    confidence: number; // 0-1
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

// =============================================================================
// UNION TYPE FOR ALL EVENTS
// =============================================================================

export type AnySherlockEvent =
    | SherlockStartEvent
    | SherlockThinkingEvent
    | SherlockSessionCreatedEvent
    | SherlockSessionResumedEvent
    | SherlockToolCallEvent
    | SherlockToolResultEvent
    | SherlockDataChangedEvent
    | SherlockUICommandEvent
    | SherlockTextEvent
    | SherlockCompleteEvent
    | SherlockErrorEvent
    | ArtifactCreatedEvent
    | ArtifactUpdatedEvent
    | UIActionEvent;

// =============================================================================
// REQUEST TYPES
// =============================================================================

export interface InvestigateRequest {
    case_id: string;
    query: string;
    session_id?: string;
    mode?: 'investigate';
    proceed_with_plan?: boolean;
    user_id?: string;
    model?: 'sonnet' | 'opus';
}

// =============================================================================
// UI STATE TYPES
// =============================================================================

/** Tool call with its result (for display) */
export interface ToolCallWithResult {
    id: string;
    tool: string;
    description: string;
    parameters?: Record<string, any>;
    status: 'pending' | 'running' | 'complete' | 'failed';
    result?: {
        success: boolean;
        output: string;
    };
    startTime: number;
    endTime?: number;
}

/** Message in the Sherlock conversation */
export interface SherlockMessage {
    id: string;
    role: 'user' | 'assistant' | 'system';
    content: string;
    timestamp: string;
    toolCalls?: ToolCallWithResult[];
    isStreaming?: boolean;
}

/** Full Sherlock state for the store */
export interface SherlockState {
    // Session management
    sessionId: string | null;
    sandboxId: string | null;
    currentCaseId: string | null;

    // Investigation status
    isInvestigating: boolean;
    isConnecting: boolean;

    // Tool call tracking (by tool_call_id)
    pendingToolCalls: Map<string, ToolCallWithResult>;

    // Conversation
    messages: SherlockMessage[];
    currentStreamingText: string;

    // Error state
    error: string | null;

    // Duration tracking
    investigationStartTime: number | null;
    lastDuration: number | null;
}

// =============================================================================
// TOOL ICON MAPPING
// =============================================================================

export type ToolType =
    | 'search'
    | 'read_file'
    | 'ls'
    | 'bash'
    | 'execute'
    | 'python'
    | 'write_file'
    | 'entity'
    | 'relationship'
    | 'event'
    | 'location'
    | 'lead'
    | 'hypothesis'
    | 'citation'
    | 'ui_action'
    | 'Task'
    | 'grep'
    | 'find'
    | 'default';

/** Maps tool names to their display configuration */
export interface ToolDisplayConfig {
    label: string;
    iconType: ToolType;
    primaryParamKey?: string; // Which parameter to show prominently
    color: string; // Accent color for the tool
}

export const TOOL_DISPLAY_CONFIG: Record<string, ToolDisplayConfig> = {
    // Consolidated CRUD tools
    entity: {
        label: 'Managing Entity',
        iconType: 'entity',
        primaryParamKey: 'name',
        color: '#06B6D4', // Cyan
    },
    relationship: {
        label: 'Managing Relationship',
        iconType: 'relationship',
        primaryParamKey: 'description',
        color: '#3B82F6', // Blue
    },
    event: {
        label: 'Managing Event',
        iconType: 'event',
        primaryParamKey: 'name',
        color: '#8B5CF6', // Purple
    },
    location: {
        label: 'Managing Location',
        iconType: 'location',
        primaryParamKey: 'name',
        color: '#10B981', // Emerald
    },
    lead: {
        label: 'Managing Lead',
        iconType: 'lead',
        primaryParamKey: 'title',
        color: '#EF4444', // Red
    },
    hypothesis: {
        label: 'Managing Hypothesis',
        iconType: 'hypothesis',
        primaryParamKey: 'statement',
        color: '#F97316', // Orange
    },
    citation: {
        label: 'Managing Citation',
        iconType: 'citation',
        primaryParamKey: 'text',
        color: '#EC4899', // Pink
    },
    ui_action: {
        label: 'UI Action',
        iconType: 'ui_action',
        primaryParamKey: 'action',
        color: '#2563EB', // Blue-600
    },
    // Code execution tools
    search: {
        label: 'Searching Evidence',
        iconType: 'search',
        primaryParamKey: 'query',
        color: '#3B82F6', // Blue
    },
    read_file: {
        label: 'Reading File',
        iconType: 'read_file',
        primaryParamKey: 'path',
        color: '#8B5CF6', // Purple
    },
    ls: {
        label: 'Listing Directory',
        iconType: 'ls',
        primaryParamKey: 'path',
        color: '#6366F1', // Indigo
    },
    bash: {
        label: 'Executing Command',
        iconType: 'bash',
        primaryParamKey: 'command',
        color: '#10B981', // Emerald
    },
    execute: {
        label: 'Executing Code',
        iconType: 'execute',
        primaryParamKey: 'code',
        color: '#10B981', // Emerald
    },
    python: {
        label: 'Running Python',
        iconType: 'python',
        primaryParamKey: 'code',
        color: '#F59E0B', // Amber
    },
    write_file: {
        label: 'Writing File',
        iconType: 'write_file',
        primaryParamKey: 'path',
        color: '#EC4899', // Pink
    },
    Task: {
        label: 'Delegating Task',
        iconType: 'Task',
        primaryParamKey: 'description',
        color: '#14B8A6', // Teal
    },
    grep: {
        label: 'Pattern Matching',
        iconType: 'grep',
        primaryParamKey: 'pattern',
        color: '#84CC16', // Lime
    },
    find: {
        label: 'Finding Files',
        iconType: 'find',
        primaryParamKey: 'name',
        color: '#22D3EE', // Cyan
    },
    // Legacy tool name aliases
    entity_create: { label: 'Creating Entity', iconType: 'entity', primaryParamKey: 'name', color: '#06B6D4' },
    entity_update: { label: 'Updating Entity', iconType: 'entity', primaryParamKey: 'name', color: '#06B6D4' },
    entity_delete: { label: 'Deleting Entity', iconType: 'entity', primaryParamKey: 'name', color: '#06B6D4' },
    entity_search: { label: 'Searching Entities', iconType: 'entity', primaryParamKey: 'query', color: '#06B6D4' },
    relationship_create: { label: 'Creating Relationship', iconType: 'relationship', primaryParamKey: 'description', color: '#3B82F6' },
    relationship_update: { label: 'Updating Relationship', iconType: 'relationship', primaryParamKey: 'description', color: '#3B82F6' },
    relationship_delete: { label: 'Deleting Relationship', iconType: 'relationship', primaryParamKey: 'description', color: '#3B82F6' },
    event_create: { label: 'Creating Event', iconType: 'event', primaryParamKey: 'name', color: '#8B5CF6' },
    event_update: { label: 'Updating Event', iconType: 'event', primaryParamKey: 'name', color: '#8B5CF6' },
    event_delete: { label: 'Deleting Event', iconType: 'event', primaryParamKey: 'name', color: '#8B5CF6' },
    location_create: { label: 'Creating Location', iconType: 'location', primaryParamKey: 'name', color: '#10B981' },
    location_update: { label: 'Updating Location', iconType: 'location', primaryParamKey: 'name', color: '#10B981' },
    location_delete: { label: 'Deleting Location', iconType: 'location', primaryParamKey: 'name', color: '#10B981' },
    lead_create: { label: 'Creating Lead', iconType: 'lead', primaryParamKey: 'title', color: '#EF4444' },
    lead_update: { label: 'Updating Lead', iconType: 'lead', primaryParamKey: 'title', color: '#EF4444' },
    lead_close: { label: 'Closing Lead', iconType: 'lead', primaryParamKey: 'title', color: '#EF4444' },
    hypothesis_create: { label: 'Creating Hypothesis', iconType: 'hypothesis', primaryParamKey: 'statement', color: '#F97316' },
    hypothesis_add_evidence: { label: 'Adding Evidence to Hypothesis', iconType: 'hypothesis', primaryParamKey: 'statement', color: '#F97316' },
    hypothesis_resolve: { label: 'Resolving Hypothesis', iconType: 'hypothesis', primaryParamKey: 'statement', color: '#F97316' },
    citation_create: { label: 'Creating Citation', iconType: 'citation', primaryParamKey: 'text', color: '#EC4899' },
};

export function getToolDisplayConfig(toolName: string): ToolDisplayConfig {
    return TOOL_DISPLAY_CONFIG[toolName] || {
        label: toolName.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
        iconType: 'default',
        color: '#71717A', // Zinc
    };
}
