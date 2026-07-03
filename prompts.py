# ============================================================================
# FrozenLake reflection system prompt (single source of truth).
# ============================================================================

# VPRL paper's exact published Direct text baseline prompt. Uses an UPPERCASE
# <ANSWER> tag and SPACE-separated actions (not comma-separated).
SYSTEM_PROMPT_TAG_VPRL_DIRECT = """\
Task: Frozen Lake Shortest Path Planning
You are given an image of a grid-based environment. In this environment:
- An elf marks the starting position.
- A gift represents the goal.
- Some cells contain ice holes that are impassable for the elf.
- The elf can move in one of four directions only: "up", "down", "left",
or "right". Each move transitions the elf by one cell in the
corresponding absolute direction. Diagonal movement is not permitted.
Your task is to analyze the image and generate the shortest valid
sequence of actions that moves the elf from the starting position to
the goal without stepping into any ice holes.
Provide your final answer enclosed between <ANSWER> and </ANSWER>, for
example: <ANSWER>right up up</ANSWER>."""
