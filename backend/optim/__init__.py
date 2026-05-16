"""Optimization layer — currently houses the driver daily-plan DP solver.

Designed to be a pure-Python, dependency-light home for any *deterministic*
optimization or scoring logic that sits between the data layer and the agents.
Agents call into here via thin tool wrappers in backend/tools/*.
"""
