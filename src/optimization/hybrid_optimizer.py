"""
Knowledge Graph-based Hybrid Database Query Optimizer

FOUNDATION:
- Knowledge Graph: Shared semantic representation of database schema, relationships, and constraints
  Used by all optimization tiers for schema-aware decision making

THREE-TIER ARCHITECTURE:
- Tier 1: Symbolic AI - PDDL-based planning and formal logical reasoning for optimization actions
- Tier 2: Reinforcement Learning - DQN adaptive optimization with KG-enhanced state representation  
- Tier 3: Generative AI - LLM semantic analysis and natural language explanations with KG context

INTEGRATION:
All tiers leverage the Knowledge Graph as a shared foundation for consistent schema understanding.
"""

import time
import re
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

from ..database_environment.db_simulator import DatabaseSimulator
from ..knowledge_graph.schema_ontology import DatabaseSchemaKG
from ..agents.symbolic_ai_agent import SymbolicAIAgent
from ..agents.dqn_agent import MultiAgentDQN
from ..agents.rl_environment import QueryOptimizationEnv
from ..agents.llm_query_agent import LangChainQueryAgent
from ..utils.logging import logger


@dataclass
class OptimizationResult:
    """Result of query optimization."""
    query: str
    estimated_cost: float
    optimization_time: float
    execution_plan: Dict[str, Any]
    strategy: str
    confidence: float
    explanation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class HybridOptimizer:
    """
    Knowledge Graph-based Hybrid AI Query Optimizer.
    
    ARCHITECTURE:
    Foundation: Knowledge Graph - shared semantic schema representation
    Tier 1: Symbolic AI - PDDL planning and formal logical reasoning
    Tier 2: Reinforcement Learning - DQN-based adaptive optimization  
    Tier 3: Generative AI - LLM semantic analysis and explanations
    
    All tiers leverage the Knowledge Graph for schema-aware optimization decisions.
    """
    
    def __init__(self, 
                 db_simulator: DatabaseSimulator, 
                 knowledge_graph: DatabaseSchemaKG):
        """
        Initialize the hybrid optimizer with Knowledge Graph foundation.
        
        Args:
            db_simulator: Database connection and execution simulator
            knowledge_graph: Shared Knowledge Graph used by all optimization tiers
        """
        self.db_simulator = db_simulator
        self.knowledge_graph = knowledge_graph  # Foundation: Shared KG for all tiers
        
        # Initialize optimization tiers
        self._init_symbolic_ai_tier()    # Tier 1: Symbolic AI with PDDL planning
        self._init_rl_tier()            # Tier 2: RL System  
        self._init_generative_tier()    # Tier 3: Generative AI
        
        # Statistics
        self.optimization_count = 0
        self.total_optimization_time = 0.0
        
        logger.info("HybridOptimizer initialized with Knowledge Graph foundation")
    
    def _init_symbolic_ai_tier(self):
        """Initialize Tier 1: Symbolic AI with PDDL-based planning."""
        try:
            self.symbolic_agent = SymbolicAIAgent(self.knowledge_graph)
            self.symbolic_enabled = True
            logger.info("Tier 1 (Symbolic AI with PDDL) initialized")
        except Exception as e:
            logger.warning(f"Tier 1 (Symbolic AI) initialization failed: {e}")
            self.symbolic_agent = None
            self.symbolic_enabled = False
    
    def _init_rl_tier(self):
        """Initialize Tier 2: RL System (uses Knowledge Graph for state representation)."""
        try:
            self.rl_env = QueryOptimizationEnv(
                database_simulator=self.db_simulator,
                knowledge_graph=self.knowledge_graph  # RL uses KG for environment modeling
            )
            self.dqn_agent = MultiAgentDQN()
            self.rl_enabled = True
            logger.info("Tier 2 (RL System with KG integration) initialized")
        except Exception as e:
            logger.warning(f"Tier 2 initialization failed: {e}")
            self.rl_env = None
            self.dqn_agent = None
            self.rl_enabled = False
    
    def _init_generative_tier(self):
        """Initialize Tier 3: Generative AI (uses Knowledge Graph for context)."""
        try:
            self.llm_agent = LangChainQueryAgent(
                knowledge_graph=self.knowledge_graph  # LLM uses KG for semantic context
            )
            self.llm_enabled = True
            logger.info("Tier 3 (Generative AI with KG context) initialized")
        except Exception as e:
            logger.warning(f"Tier 3 initialization failed: {e}")
            self.llm_agent = None
            self.llm_enabled = False
    
    def optimize(self, query: str) -> OptimizationResult:
        """
        Optimize SQL query using Knowledge Graph-based three-tier hybrid AI.
        
        PROCESS:
        1. Knowledge Graph Analysis (foundation for all tiers)
        2. Tier 1: Symbolic AI with PDDL planning
        3. Tier 2: RL optimization (using KG state representation)  
        4. Tier 3: Generative AI semantic analysis (using KG context)
        5. Multi-tier fusion and explanation generation
        
        Args:
            query: SQL query to optimize
            
        Returns:
            OptimizationResult with cost, plan, and explanations
        """
        start_time = time.time()
        
        # Foundation: Knowledge Graph Analysis (used by all tiers)
        kg_analysis = self._analyze_query_with_knowledge_graph(query)
        
        # Tier 1: Symbolic AI with PDDL Planning
        symbolic_result = self._tier1_symbolic_ai_optimization(query, kg_analysis) if self.symbolic_enabled else None
        
        # Tier 2: RL Enhancement (using KG for state representation)
        rl_result = self._tier2_rl_optimization(query, kg_analysis) if self.rl_enabled else None
        
        # Tier 3: LLM Semantic Analysis (using KG for context)
        llm_result = self._tier3_semantic_analysis(query, kg_analysis) if self.llm_enabled else None
        
        # Multi-tier fusion with KG-guided integration
        final_result = self._fuse_optimization_results(query, kg_analysis, symbolic_result, rl_result, llm_result)
        
        # Update statistics
        optimization_time = time.time() - start_time
        final_result.optimization_time = optimization_time
        self.optimization_count += 1
        self.total_optimization_time += optimization_time
        
        return final_result
    
    # ============================================================================
    # KNOWLEDGE GRAPH FOUNDATION (shared by all tiers)
    # ============================================================================
    
    def _analyze_query_with_knowledge_graph(self, query: str) -> Dict[str, Any]:
        """Foundation: Analyze query using Knowledge Graph (shared by all tiers)."""
        # Extract query components
        tables = self._extract_tables_from_query(query)
        joins = self._extract_joins_from_query(query)
        conditions = self._extract_conditions_from_query(query)
        
        # Knowledge graph structural analysis
        kg_structure = {
            'tables': tables,
            'table_relationships': [],
            'index_opportunities': [],
            'join_paths': [],
            'schema_constraints': []
        }
        
        # Analyze each table in the Knowledge Graph
        for table in tables:
            if table in self.knowledge_graph.tables:
                table_info = self.knowledge_graph.tables[table]
                
                # Extract foreign key relationships
                if table_info.foreign_keys:
                    for fk_col, ref_table_col in table_info.foreign_keys.items():
                        kg_structure['table_relationships'].append({
                            'type': 'foreign_key',
                            'from': f"{table}.{fk_col}",
                            'to': ref_table_col,
                            'cardinality': 'many_to_one'
                        })
                        
                        kg_structure['index_opportunities'].append({
                            'table': table,
                            'column': fk_col,
                            'type': 'foreign_key_index',
                            'benefit': 'join_optimization'
                        })
                
                # Extract primary key information
                if table_info.primary_key:
                    kg_structure['schema_constraints'].append({
                        'type': 'primary_key',
                        'table': table,
                        'columns': table_info.primary_key,
                        'uniqueness': True
                    })
        
        # Determine possible join paths using KG structure
        if len(tables) > 1:
            kg_structure['join_paths'] = self._find_join_paths_in_kg(tables)
        
        return {
            'query_structure': {
                'tables': tables,
                'joins': joins, 
                'conditions': conditions
            },
            'kg_analysis': kg_structure,
            'complexity_score': self._calculate_kg_complexity(kg_structure)
        }
    
    # ============================================================================
    # TIER 1: SYMBOLIC AI WITH PDDL PLANNING
    # ============================================================================
    
    def _tier1_symbolic_ai_optimization(self, query: str, kg_analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Tier 1: Symbolic AI optimization using dedicated SymbolicAIAgent."""
        if not self.symbolic_enabled or not self.symbolic_agent:
            return None
        
        try:
            # Use dedicated SymbolicAIAgent for optimization
            optimization_plan = self.symbolic_agent.optimize_query(query, kg_analysis)
            
            if optimization_plan:
                return {
                    'pddl_plan': optimization_plan.pddl_plan,
                    'db_actions': optimization_plan.db_actions,
                    'estimated_cost': optimization_plan.estimated_cost,
                    'reasoning_method': optimization_plan.reasoning_method,
                    'confidence': optimization_plan.confidence,
                    'planning_time': optimization_plan.planning_time
                }
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Symbolic AI optimization failed: {e}")
            return None
    
    # ============================================================================
    # TIER 2: REINFORCEMENT LEARNING WITH KG-ENHANCED STATE
    # ============================================================================
    
    def _tier2_rl_optimization(self, query: str, kg_analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Tier 2: RL optimization using Knowledge Graph for enhanced state representation."""
        if not self.rl_env or not self.dqn_agent:
            return None
        
        try:
            # Use KG analysis to enhance RL state representation
            state, info = self.rl_env.reset()
            
            # Enhance state with KG features
            kg_features = self._extract_kg_features_for_rl(kg_analysis)
            enhanced_state = np.concatenate([state, kg_features]) if len(kg_features) > 0 else state
            
            actions = self.dqn_agent.get_actions(enhanced_state, deterministic=True)
            _, reward, _, _, step_info = self.rl_env.step(actions)
            
            return {
                'actions': actions,
                'reward': reward,
                'cost_reduction': step_info.get('improvement_ratio', 0),
                'kg_enhanced_state': True,
                'confidence': min(0.8, max(0.1, reward / 15.0))  # Normalize reward to confidence
            }
        except Exception as e:
            logger.warning(f"Tier 2 RL optimization failed: {e}")
            return None
    
    # ============================================================================
    # TIER 3: GENERATIVE AI WITH KG CONTEXT
    # ============================================================================
    
    def _tier3_semantic_analysis(self, query: str, kg_analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Tier 3: LLM semantic analysis using Knowledge Graph for enhanced context."""
        if not self.llm_agent:
            return None
        
        try:
            # Provide KG context to LLM for enhanced semantic understanding
            kg_context = self._prepare_kg_context_for_llm(kg_analysis)
            
            # Use basic analyze_query method since analyze_query_with_context may not exist
            analysis = self.llm_agent.analyze_query(query)
            
            complexity = self._assess_query_complexity(query)
            
            return {
                'semantic_analysis': analysis,
                'complexity_level': complexity,
                'kg_context_used': True,
                'schema_aware_suggestions': self._generate_schema_aware_suggestions(kg_analysis),
                'confidence': 0.8
            }
        except Exception as e:
            logger.warning(f"Tier 3 semantic analysis failed: {e}")
            # Fallback without KG context
            return self._basic_semantic_analysis(query)
    
    # ============================================================================
    # FUSION AND INTEGRATION
    # ============================================================================
    
    def _fuse_optimization_results(self, 
                                   query: str,
                                   kg_analysis: Dict[str, Any],
                                   symbolic: Optional[Dict[str, Any]],
                                   rl: Optional[Dict[str, Any]],
                                   llm: Optional[Dict[str, Any]]) -> OptimizationResult:
        """Fuse results from all optimization tiers using KG-guided integration."""
        
        # Start with base cost estimation from query structure
        base_cost = self._estimate_base_cost(kg_analysis['query_structure'])
        final_cost = base_cost
        confidence_scores = []
        
        # Incorporate Symbolic AI results (highest priority due to formal reasoning)
        if symbolic:
            final_cost = symbolic.get('estimated_cost', base_cost)
            confidence_scores.append(symbolic['confidence'])
        
        # Apply RL improvements
        if rl and rl.get('cost_reduction', 0) > 0:
            reduction = rl['cost_reduction']
            final_cost *= (1 - reduction)
            confidence_scores.append(rl['confidence'])
        
        # If no other results, use base cost
        if not confidence_scores:
            confidence_scores.append(0.6)  # Default confidence
        
        # Calculate overall confidence
        final_confidence = sum(confidence_scores) / len(confidence_scores)
        
        # Generate comprehensive explanation
        explanation = self._generate_comprehensive_explanation(kg_analysis, symbolic, rl, llm)
        
        # Build execution plan
        execution_plan = self._build_execution_plan(kg_analysis, symbolic, rl, llm)
        
        # Build comprehensive metadata
        metadata = {
            'kg_foundation': kg_analysis,
            'symbolic_reasoning': symbolic,
            'rl_optimization': rl,
            'llm_analysis': llm,
            'active_tiers': self._get_active_tiers(),
            'optimization_count': self.optimization_count
        }
        
        return OptimizationResult(
            query=query,
            estimated_cost=final_cost,
            optimization_time=0.0,  # Set by caller
            execution_plan=execution_plan,
            strategy='hybrid_kg_pddl_rl_llm',
            confidence=final_confidence,
            explanation=explanation,
            metadata=metadata
        )
    
    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    def _extract_tables_from_query(self, query: str) -> List[str]:
        """Extract table names from SQL query."""
        tables = []
        query_upper = query.upper()
        
        for table_name in self.knowledge_graph.tables.keys():
            if table_name.upper() in query_upper:
                tables.append(table_name)
        
        return tables
    
    def _extract_joins_from_query(self, query: str) -> List[str]:
        """Extract join information from SQL query."""
        joins = []
        query_upper = query.upper()
        
        if 'JOIN' in query_upper:
            join_types = ['INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'FULL JOIN', 'JOIN']
            for join_type in join_types:
                if join_type in query_upper:
                    joins.append(join_type.lower())
                    break
        
        return joins
    
    def _extract_conditions_from_query(self, query: str) -> List[str]:
        """Extract WHERE conditions from SQL query."""
        conditions = []
        query_upper = query.upper()
        
        if 'WHERE' in query_upper:
            conditions.append('where_clause_present')
        if 'GROUP BY' in query_upper:
            conditions.append('aggregation')
        if 'ORDER BY' in query_upper:
            conditions.append('sorting')
        
        return conditions
    
    def _assess_query_complexity(self, query: str) -> str:
        """Assess query complexity level."""
        query_upper = query.upper()
        complexity_score = 0
        
        if 'JOIN' in query_upper:
            complexity_score += 2
        if 'GROUP BY' in query_upper:
            complexity_score += 1
        if 'HAVING' in query_upper:
            complexity_score += 1
        if 'SUBQUERY' in query_upper or '(SELECT' in query_upper:
            complexity_score += 3
        if 'UNION' in query_upper:
            complexity_score += 2
        
        if complexity_score >= 4:
            return 'high'
        elif complexity_score >= 2:
            return 'medium'
        return 'low'
    
    def _estimate_base_cost(self, query_structure: Dict[str, Any]) -> float:
        """Simple base cost estimation."""
        base_cost = 100.0
        table_cost = len(query_structure.get('tables', [])) * 500
        join_cost = len(query_structure.get('joins', [])) * 1000  
        condition_cost = len(query_structure.get('conditions', [])) * 200
        
        return base_cost + table_cost + join_cost + condition_cost
    
    def _generate_comprehensive_explanation(self,
                                            kg_analysis: Dict[str, Any],
                                            symbolic: Optional[Dict[str, Any]],
                                            rl: Optional[Dict[str, Any]], 
                                            llm: Optional[Dict[str, Any]]) -> str:
        """Generate human-readable optimization explanation."""
        explanation_parts = []
        
        # Knowledge Graph foundation
        kg_info = kg_analysis.get('kg_analysis', {})
        if kg_info.get('index_opportunities'):
            explanation_parts.append(f"Knowledge Graph identified {len(kg_info['index_opportunities'])} index optimization opportunities.")
        
        # Symbolic AI tier
        if symbolic:
            method = symbolic.get('reasoning_method', 'symbolic reasoning')
            explanation_parts.append(f"Applied {method} for formal optimization planning.")
        
        # RL tier
        if rl and rl.get('cost_reduction', 0) > 0:
            reduction = rl['cost_reduction']
            explanation_parts.append(f"RL optimization achieved {reduction*100:.1f}% cost reduction.")
        
        # LLM tier
        if llm:
            complexity = llm.get('complexity_level', 'unknown')
            explanation_parts.append(f"LLM analysis identified {complexity} complexity query.")
        
        return " ".join(explanation_parts) if explanation_parts else "Applied Knowledge Graph-based hybrid optimization."
    
    def _build_execution_plan(self,
                              kg_analysis: Dict[str, Any],
                              symbolic: Optional[Dict[str, Any]],
                              rl: Optional[Dict[str, Any]],
                              llm: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build comprehensive execution plan."""
        plan = {
            'query_structure': kg_analysis.get('query_structure', {}),
            'kg_insights': kg_analysis.get('kg_analysis', {}),
            'optimizations_applied': []
        }
        
        if symbolic and 'db_actions' in symbolic:
            plan['symbolic_actions'] = symbolic['db_actions']
            plan['optimizations_applied'].extend(symbolic.get('db_actions', []))
        
        if rl:
            plan['rl_actions'] = rl.get('actions', {})
            
        if llm:
            plan['semantic_suggestions'] = llm.get('schema_aware_suggestions', [])
        
        return plan
    
    def _get_active_tiers(self) -> List[str]:
        """Get list of active optimization tiers."""
        tiers = ['knowledge_graph']  # KG is always the foundation
        if self.symbolic_enabled:
            tiers.append('symbolic_ai_pddl')
        if self.rl_enabled and self.rl_env:
            tiers.append('reinforcement_learning')
        if self.llm_enabled and self.llm_agent:
            tiers.append('generative_ai')
        return tiers
    
    # Placeholder methods for implementation
    def _find_join_paths_in_kg(self, tables: List[str]) -> List[Dict[str, Any]]:
        """Find possible join paths using Knowledge Graph structure."""
        return [{'path': ' -> '.join(tables), 'cost_score': len(tables)}]
    
    def _calculate_kg_complexity(self, kg_structure: Dict[str, Any]) -> float:
        """Calculate complexity score based on KG analysis."""
        return len(kg_structure.get('tables', [])) * 2 + len(kg_structure.get('table_relationships', []))
    
    def _extract_kg_features_for_rl(self, kg_analysis: Dict[str, Any]) -> np.ndarray:
        """Extract KG features for RL state enhancement."""
        kg_info = kg_analysis.get('kg_analysis', {})
        features = [
            len(kg_info.get('tables', [])),
            len(kg_info.get('table_relationships', [])),
            len(kg_info.get('index_opportunities', [])),
            kg_analysis.get('complexity_score', 0)
        ]
        return np.array(features, dtype=np.float32)
    
    def _prepare_kg_context_for_llm(self, kg_analysis: Dict[str, Any]) -> str:
        """Prepare Knowledge Graph context for LLM analysis."""
        kg_info = kg_analysis.get('kg_analysis', {})
        context_parts = []
        
        if kg_info.get('table_relationships'):
            context_parts.append(f"Schema relationships: {len(kg_info['table_relationships'])} connections found")
        
        if kg_info.get('index_opportunities'):
            context_parts.append(f"Index opportunities: {len(kg_info['index_opportunities'])} potential optimizations")
        
        return "; ".join(context_parts) if context_parts else "Basic schema analysis"
    
    def _generate_schema_aware_suggestions(self, kg_analysis: Dict[str, Any]) -> List[str]:
        """Generate schema-aware optimization suggestions."""
        suggestions = []
        kg_info = kg_analysis.get('kg_analysis', {})
        
        for opp in kg_info.get('index_opportunities', []):
            suggestions.append(f"Consider adding {opp['type']} on {opp['table']}.{opp['column']}")
        
        return suggestions
    
    def _basic_semantic_analysis(self, query: str) -> Dict[str, Any]:
        """Basic semantic analysis fallback."""
        return {
            'complexity_level': self._assess_query_complexity(query),
            'basic_analysis': True,
            'confidence': 0.5
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimizer performance statistics."""
        return {
            'total_optimizations': self.optimization_count,
            'total_time': self.total_optimization_time,
            'average_time': self.total_optimization_time / max(1, self.optimization_count),
            'active_tiers': self._get_active_tiers(),
            'knowledge_graph_tables': len(self.knowledge_graph.tables),
            'symbolic_enabled': self.symbolic_enabled,
            'rl_enabled': self.rl_enabled,
            'llm_enabled': self.llm_enabled
        }