# Intelligent Database Query Optimizer

A research project exploring AI-driven database query optimization using knowledge graphs and multi-agent systems.

## Overview

Database query optimization is mostly rule-based and static. Current optimizers can't adapt to changing data patterns or explain their decisions. This project explores a hybrid AI approach that combines knowledge graphs, LLM agents, and reinforcement learning for smarter query optimization.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Database Schema │    │ LLM Query       │    │ Rule-Based      │
│ Knowledge Graph │◄──►│ Understanding   │◄──►│ Optimizer       │
│ (SQLite/PgSQL)  │    │ Agent           │    │ (Baseline)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                       ▲                       ▲
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│         Database Execution Environment                          │
│      (PostgreSQL/SQLite + Performance Monitoring)              │
└─────────────────────────────────────────────────────────────────┘
```

## Components

| Component | Technology |
|-----------|------------|
| Knowledge Graph | Python + Database Metadata |
| LLM Agent | LangChain + HuggingFace |
| Rule-Based Optimizer | Python Heuristics |
| Database Environment | PostgreSQL + SQLite |
| RL Agents | PyTorch (planned) | 🚧 Future |
| PDDL Planning | (planned) | 🚧 Future |

## Installation

```bash
git clone https://github.com/yourusername/intelligent-db-optimizer.git
cd intelligent-db-optimizer
pip install -r requirements.txt
pip install psycopg2-binary  # for PostgreSQL
```

## Quick Start

```python
from src.knowledge_graph.schema_ontology import DatabaseSchemaKG
from src.database_environment.db_simulator import DatabaseSimulator

# Initialize system
db = DatabaseSimulator(db_type="postgresql")
db.connect()
db.create_sample_tables()

# Build knowledge graph
kg = DatabaseSchemaKG(db_type="postgresql")
kg.build_from_database(db.connection)

# Get join order suggestions
tables = ["customers", "orders", "lineitem"]
optimal_order = kg.suggest_join_order(tables)
print(f"Suggested join order: {' -> '.join(optimal_order)}")
```

## Current Status

The system includes:
- Database schema extraction and knowledge graph representation
- Multi-engine support (PostgreSQL and SQLite)
- LLM-based query understanding using LangChain
- Rule-based optimization baseline
- Basic join order optimization

Future work will add reinforcement learning agents and automated planning.

## Testing

```bash
python test_knowledge_graph.py
python tests/unit_tests/test_postgresql_connection.py
```

## Project Structure

```
intelligent-db-optimizer/
├── src/
│   ├── database_environment/    # Database simulators
│   ├── knowledge_graph/         # Schema representation
│   ├── agents/                  # LLM agents
│   └── utils/                   # Config and logging
├── evaluation/
│   ├── baselines/              # Rule-based optimizer
│   └── benchmarks/             # TPC-H, TPC-DS (planned)
├── tests/
└── results/                    # Experiment results
```

## Evaluation Plan

The system will be evaluated using:
- **Benchmarks**: TPC-H, TPC-DS, Join Order Benchmark
- **Baselines**: PostgreSQL optimizer, rule-based approaches
- **Metrics**: Query execution time, throughput, resource usage
- **Success criteria**: >20% improvement over baseline optimizers
