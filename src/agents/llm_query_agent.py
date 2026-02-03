"""
LangChain-based query understanding and optimization agent.
"""

from typing import Dict, List, Any, Optional
import os
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.llms.base import LLM
from langchain.schema import BaseMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.knowledge_graph.schema_ontology import DatabaseSchemaKG
from src.utils.config import config
from src.utils.logging import logger

class SimpleLLM(LLM):
    """Simple fallback LLM for when external services are unavailable."""
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Rule-based responses for database optimization."""
        prompt_lower = prompt.lower()
        
        if "join order" in prompt_lower or "join" in prompt_lower:
            return """For optimal join ordering:
1. Start with the smallest table to reduce intermediate results
2. Use foreign key relationships when available
3. Apply filters early to reduce data volume
4. Consider using EXPLAIN ANALYZE to verify execution plans"""
        
        elif "index" in prompt_lower:
            return """Index recommendations:
1. Create indexes on columns used in WHERE clauses
2. Index foreign key columns used in JOINs
3. Consider composite indexes for multi-column filters
4. Monitor index usage and remove unused indexes"""
        
        elif "optimize" in prompt_lower:
            return """SQL optimization strategies:
1. Proper join ordering (smallest table first)
2. Appropriate indexing on filter and join columns
3. Query rewriting for better performance
4. Predicate pushdown to reduce data movement
5. Use of appropriate data types and constraints"""
        
        elif "analyze" in prompt_lower:
            tables = []
            if "customers" in prompt_lower:
                tables.append("customers")
            if "orders" in prompt_lower:
                tables.append("orders")
            if "lineitem" in prompt_lower:
                tables.append("lineitem")
            
            if tables:
                return f"""Query analysis for tables {', '.join(tables)}:
- Tables involved: {len(tables)}
- Complexity: {'High' if len(tables) > 2 else 'Medium' if len(tables) == 2 else 'Low'}
- Recommendations: Ensure proper indexing on join columns, consider join order optimization"""
            
            return "This query involves standard SQL operations. Consider indexing and join optimization."
        
        else:
            return "I can help optimize SQL queries through join ordering, indexing strategies, and query rewriting techniques."
    
    @property
    def _llm_type(self) -> str:
        return "simple"


class LLMFactory:
    """Factory for creating different LLM instances."""
    
    @staticmethod
    def create_llm(provider: str = None) -> LLM:
        """Create LLM instance based on provider."""
        provider = provider or config.llm.default_provider
        
        try:
            if provider == "ollama":
                return Ollama(
                    base_url=config.llm.ollama_base_url,
                    model=config.llm.ollama_model,
                    temperature=config.llm.temperature
                )
            
            elif provider == "openai":
                if not config.llm.openai_api_key:
                    logger.warning("OpenAI API key not found, falling back to simple LLM")
                    return SimpleLLM()
                
                return OpenAI(
                    api_key=config.llm.openai_api_key,
                    model=config.llm.openai_model,
                    temperature=config.llm.temperature,
                    max_tokens=config.llm.max_tokens
                )
            
            elif provider == "simple":
                return SimpleLLM()
            
            else:
                logger.warning(f"Unknown LLM provider: {provider}, using simple LLM")
                return SimpleLLM()
                
        except Exception as e:
            logger.warning(f"Failed to initialize {provider} LLM: {e}, falling back to simple LLM")
            return SimpleLLM()
        

class LangChainQueryAgent:
    """
    Enhanced LangChain-based SQL query understanding and optimization agent.
    """
    
    def __init__(self, knowledge_graph: DatabaseSchemaKG, llm_provider: str = None):
        self.kg = knowledge_graph
        
        # Initialize LLM
        self.llm = LLMFactory.create_llm(llm_provider)
        logger.info(f"Initialized LLM: {self.llm._llm_type}")
        
        # Initialize embeddings for semantic similarity
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Create modern prompt templates (fixing deprecation warnings)
        self.analysis_prompt = PromptTemplate(
            input_variables=["query", "schema_info", "table_stats"],
            template="""You are a database query analysis expert.

Query to analyze: {query}

Database Schema: {schema_info}

Table Statistics: {table_stats}

Please analyze this query and identify:
1. Tables involved and their relationships
2. Query complexity level (Low/Medium/High)
3. Join patterns and filtering conditions
4. Potential performance bottlenecks
5. Optimization opportunities

Provide a concise analysis focused on actionable insights."""
        )

        self.optimization_prompt = PromptTemplate(
            input_variables=["query", "schema_info", "table_stats", "analysis"],
            template="""You are a database optimization expert.

Query to optimize: {query}

Database Schema: {schema_info}
Table Statistics: {table_stats}
Previous Analysis: {analysis}

Provide specific optimization recommendations:
1. Join order suggestions with reasoning
2. Index recommendations for better performance
3. Query rewriting opportunities
4. Expected performance improvement

Focus on practical, implementable suggestions."""
        )
        
        # Create runnable chains (modern LangChain approach)
        self.analysis_chain = self.analysis_prompt | self.llm
        self.optimization_chain = self.optimization_prompt | self.llm
        
        logger.info("LangChain Query Agent initialized successfully")

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query using LangChain and knowledge graph."""
        try:
            # Get schema and statistics
            schema_info = self._get_schema_summary()
            table_stats = self._get_table_statistics()
            
            # Use modern chain invoke method
            llm_analysis = self.analysis_chain.invoke({
                "query": query,
                "schema_info": schema_info,
                "table_stats": table_stats
            })
            
            # Knowledge graph analysis
            kg_analysis = self._analyze_with_knowledge_graph(query)
            
            return {
                "query": query,
                "llm_analysis": llm_analysis,
                "knowledge_graph_analysis": kg_analysis,
                "schema_info": schema_info,
                "table_stats": table_stats,
                "llm_provider": self.llm._llm_type
            }
            
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return {"error": str(e)}
        
    def optimize_query(self, query: str) -> Dict[str, Any]:
        """Generate optimized query using LangChain."""
        try:
            # First analyze the query
            analysis_result = self.analyze_query(query)
            
            if "error" in analysis_result:
                return analysis_result
            
            # Generate optimization recommendations
            optimization_result = self.optimization_chain.invoke({
                "query": query,
                "schema_info": analysis_result["schema_info"],
                "table_stats": analysis_result["table_stats"],
                "analysis": analysis_result["llm_analysis"]
            })
            
            # Get knowledge graph suggestions
            kg_suggestions = self._get_kg_optimization_suggestions(query)
            
            return {
                "original_query": query,
                "llm_optimization": optimization_result,
                "kg_suggestions": kg_suggestions,
                "optimized_query": self._generate_optimized_query(query, kg_suggestions),
                "llm_provider": self.llm._llm_type
            }
            
        except Exception as e:
            logger.error(f"Query optimization failed: {e}")
            return {"error": str(e)}
        
    def explain_optimization(self, query: str) -> str:
        """Provide detailed explanation of optimization recommendations."""
        optimization_result = self.optimize_query(query)
        
        if "error" in optimization_result:
            return f"Error in optimization: {optimization_result['error']}"
        
        explanation = f"""Query Optimization Report
{'='*50}

Original Query:
{query}

LLM Analysis & Recommendations:
{optimization_result['llm_optimization']}

Knowledge Graph Insights:
"""
        
        for suggestion in optimization_result['kg_suggestions']:
            explanation += f"- {suggestion['type'].title()}: {suggestion['suggestion']}\n"
            explanation += f"  Reasoning: {suggestion['reasoning']}\n"
        
        explanation += f"""
Optimized Query:
{optimization_result['optimized_query']}

LLM Provider: {optimization_result['llm_provider']}
"""
        
        return explanation
    
    def _get_schema_summary(self) -> str:
        """Get concise database schema summary."""
        schema_parts = []
        for table_name, table_info in self.kg.tables.items():
            key_columns = []
            if table_info.primary_key:
                key_columns.extend([f"{col}(PK)" for col in table_info.primary_key])
            if table_info.foreign_keys:
                key_columns.extend([f"{col}(FK)" for col in table_info.foreign_keys.keys()])
            
            column_info = f"{len(table_info.columns)} columns"
            if key_columns:
                column_info += f" ({', '.join(key_columns)})"
            
            schema_parts.append(f"{table_name}: {column_info}")
        
        return "; ".join(schema_parts)
    
    def _get_table_statistics(self) -> str:
        """Get table statistics summary."""
        stats_parts = []
        for table_name, table_info in self.kg.tables.items():
            stats_parts.append(f"{table_name}: {table_info.row_count} rows")
        
        return "; ".join(stats_parts)
    
    def _analyze_with_knowledge_graph(self, query: str) -> Dict[str, Any]:
        """Analyze query using knowledge graph."""
        query_upper = query.upper()
        
        # Find involved tables
        involved_tables = [
            table for table in self.kg.tables.keys()
            if table.upper() in query_upper
        ]
        
        # Suggest optimal join order
        optimal_join_order = self.kg.suggest_join_order(involved_tables) if len(involved_tables) > 1 else involved_tables
        
        # Find relationships
        relationships = []
        for rel in self.kg.relationships:
            if rel.left_table in involved_tables and rel.right_table in involved_tables:
                relationships.append({
                    "left": rel.left_table,
                    "right": rel.right_table,
                    "condition": f"{rel.left_column} = {rel.right_column}"
                })
        
        return {
            "involved_tables": involved_tables,
            "optimal_join_order": optimal_join_order,
            "relationships": relationships,
            "complexity_estimate": self._estimate_query_complexity(involved_tables, query)
        }
    
    def _estimate_query_complexity(self, tables: List[str], query: str) -> str:
        """Estimate query complexity."""
        table_count = len(tables)
        join_count = query.upper().count("JOIN")
        subquery_count = query.upper().count("SELECT") - 1
        
        if table_count <= 1 and join_count == 0:
            return "Low"
        elif table_count <= 2 and join_count <= 1 and subquery_count == 0:
            return "Medium"
        else:
            return "High"
        
    def _get_kg_optimization_suggestions(self, query: str) -> List[Dict[str, Any]]:
        """Get optimization suggestions from knowledge graph."""
        kg_analysis = self._analyze_with_knowledge_graph(query)
        suggestions = []
        
        # Join order suggestion
        if len(kg_analysis["involved_tables"]) > 1:
            suggestions.append({
                "type": "join_order",
                "suggestion": f"Optimal join order: {' -> '.join(kg_analysis['optimal_join_order'])}",
                "reasoning": "Based on table sizes and relationships in knowledge graph"
            })
        
        # Index suggestions
        for table in kg_analysis["involved_tables"]:
            table_info = self.kg.get_table_info(table)
            if table_info and table_info.row_count > 100:
                suggestions.append({
                    "type": "indexing",
                    "suggestion": f"Consider indexes on {table} join/filter columns",
                    "reasoning": f"Table has {table_info.row_count} rows, indexing would improve performance"
                })
        
        # Relationship-based suggestions
        for rel in kg_analysis["relationships"]:
            suggestions.append({
                "type": "foreign_key_optimization",
                "suggestion": f"Utilize FK relationship: {rel['left']} -> {rel['right']}",
                "reasoning": "Foreign key relationships enable efficient join execution"
            })
        
        return suggestions
    
    def _generate_optimized_query(self, original_query: str, suggestions: List[Dict[str, Any]]) -> str:
        """Generate optimized query with hints."""
        hints = []
        for suggestion in suggestions:
            hints.append(f"-- {suggestion['type'].replace('_', ' ').title()}: {suggestion['suggestion']}")
        
        if hints:
            return "\n".join(hints) + "\n" + original_query
        
        return original_query