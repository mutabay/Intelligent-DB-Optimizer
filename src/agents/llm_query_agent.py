"""
LangChain-based query understanding and optimization agent.
"""

from typing import Dict, List, Any, Optional
import os
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.base import LLM
from langchain.schema import BaseMessage
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.knowledge_graph.schema_ontology import DatabaseSchemaKG
from src.utils.logging import logger


class SimpleLLM(LLM):
    """Simple LLM wrapper for local/mock responses."""
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Simple rule-based responses for database optimization."""

        prompt_lower = prompt.lower()

        if "join order" in prompt_lower:
            return "Consider starting with the smallest table to reduce intermediate result sizes. Use EXPLAIN ANALYZE to verify the execution plan."
        
        elif "index" in prompt_lower:
            return "Create indexes on columns used in WHERE clauses and JOIN conditions. Consider composite indexes for multiple column filters."
        
        elif "optimize" in prompt_lower:
            return "Key optimizations: 1) Proper join ordering 2) Appropriate indexing 3) Query rewriting 4) Predicate pushdown"
        
        else:
            return "I can help optimize SQL queries by suggesting join orders, indexing strategies, and query rewriting techniques."
        

    @property
    def _llm_type(self) -> str:
        return "simple"
    
class LangChainQueryAgent:
    """LangChain-based SQL query understanding and optimization agent."""
    
    def __init__(self, knowledge_graph:DatabaseSchemaKG, db_connection_str:str):
        self.kg = knowledge_graph
        self.db_connection = db_connection_str

        # Create LangChain SQL database wrapper
        self.sql_db = SQLDatabase.from_uri(f"sqlite:///{db_connection_str}")

        # Initialize LLM 
        self.llm = SimpleLLM()

        # Create SQL toolkit
        self.toolkit = SQLDatabaseToolkit(db=self.sql_db, llm=self.llm)

        # Create the agent
        self.agent = create_sql_agent(
            llm=self.llm,
            toolkit=self.toolkit,
            verbose=True
        )

        # Initialize embeddings for semantic similarity
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Query optimization prompts
        self.optimization_prompt = PromptTemplate(
            input_variables=["query", "schema_info", "table_stats"],
            template="""
            You are a database query optimization expert.
            
            Query to optimize: {query}
            
            Database Schema: {schema_info}
            
            Table Statistics: {table_stats}
            
            Please provide specific optimization recommendations:
            1. Join order suggestions
            2. Index recommendations  
            3. Query rewriting opportunities
            4. Performance estimates
            
            Focus on practical, implementable suggestions.
            """
        )

        self.optimization_chain = LLMChain(
            llm=self.llm,
            prompt=self.optimization_prompt
        )

        logger.info("LangChainQueryAgent initialized successfully")

    def analyze_query(self, query: str) ->  Dict[str, Any]:
        """Analyze query using LangChain SQL agent."""
        
        try:
            # Get Schema Info
            schema_info = self._get_schema_summary()
            table_stats = self._get_table_statistics()

            # Use LangChain to analyze the query
            analysis_prompt = f"""
            Analyze this SQL query and provide insights:

            Query: {query}
            
            Please identify:
            1. Tables involved
            2. Join patterns
            3. Filtering conditions  
            4. Complexity level
            5. Potential optimization opportunities
            """

            # Get LLM Analysis
            llm_analysis = self.llm(analysis_prompt)

            # Combine with knowledge graph insights
            kg_analysis = self._analyze_with_kg(query)

            return {
                "query": query,
                "llm_analysis": llm_analysis,
                "knowledge_graph_analysis": kg_analysis,
                "schema_info": schema_info,
                "table_stats": table_stats
            }
        
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            return {"error": str(e)}
        

    def optimize_query(self, query: str) -> Dict[str, Any]:
        """Generate optimized query using LangChain."""
        try:
            schema_info = self._get_schema_summary()
            table_stats = self._get_table_statistics()

            # Use the optimization chain
            optimization_result = self.optimization_chain.run(
                query=query,
                schema_info=schema_info,
                table_stats=table_stats
            )

            # Get knowledge graph suggestions
            kg_suggestions = self._get_kg_optimization_suggestions(query)

            return {
                "original_query": query,
                "llm_optimization": optimization_result,
                "kg_suggestions": kg_suggestions,
                "optimized_query": self._generate_optimized_query(query, kg_suggestions)
            }
             
        except Exception as e:
            logger.error(f"Query optimization failed: {e}")
            return {"error": str(e)}

    def execute_with_explanation(self, query: str) -> Dict[str, Any]:
        """Execute query and provide explanation using LangChain agent."""
        try:
            # Use SQL agent to execute and explain
            result = self.agent.run(f"Execute this query and explain the results: {query}")

            return {
                "query": query,
                "agent_response": result,
                "execution_successful": True
            }
        
        except Exception as e:
            logger.error(f"Query execution with explanation failed: {e}")
            return {
                "query": query,
                "error": str(e),
                "execution_successful": False
            }
        
    def _get_schema_summary(self) -> str:
        """Get database schema summary."""
        schema_parts = []
        for table_name, table_info in self.kg.tables.items():
            schema_parts.append(f"Table {table_name}: {', '.join(table_info.columns)}")
        
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
            "relationships": relationships
        }
    
    def _get_kg_optimization_suggestions(self, query: str) -> List[Dict[str, Any]]:
        """Get optimization suggestions from knowledge graph."""
        kg_analysis = self._analyze_with_knowledge_graph(query)
        suggestions = []
        
        # Join order suggestion
        if len(kg_analysis["involved_tables"]) > 1:
            suggestions.append({
                "type": "join_order",
                "suggestion": f"Optimal join order: {' -> '.join(kg_analysis['optimal_join_order'])}",
                "reasoning": "Based on table sizes in knowledge graph"
            })
        
        # Index suggestions
        for table in kg_analysis["involved_tables"]:
            table_info = self.kg.get_table_info(table)
            if table_info and table_info.row_count > 100:
                suggestions.append({
                    "type": "indexing",
                    "suggestion": f"Consider indexes on {table} join/filter columns",
                    "reasoning": f"Table has {table_info.row_count} rows"
                })
        
        return suggestions
    
    def _generate_optimized_query(self, original_query: str, suggestions: List[Dict[str, Any]]) -> str:
        """Generate optimized query with hints."""
        optimized = original_query
        
        # Add optimization hints as comments
        hints = []
        for suggestion in suggestions:
            hints.append(f"-- {suggestion['type'].title()}: {suggestion['suggestion']}")
        
        if hints:
            optimized = "\n".join(hints) + "\n" + original_query
        
        return optimized