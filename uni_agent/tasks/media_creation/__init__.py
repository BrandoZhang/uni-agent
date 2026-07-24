"""Media-creation task: a brief -> media via the ``media-ai`` CLI + its skills.

Registered lazily via ``TASK_MODULES`` (see :mod:`uni_agent.tasks.registry`); the
concrete task + config live in :mod:`.task`, the cost-aware reward in :mod:`.reward`,
and the media-ai skills glue in :mod:`.skills`.
"""
